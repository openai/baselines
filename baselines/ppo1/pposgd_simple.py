from collections import deque
import time
import os

import tensorflow as tf
import numpy as np
from mpi4py import MPI
import cloudpickle
import joblib

from baselines.common import Dataset, explained_variance, fmt_row, zipsame, BaseRLModel, set_global_seeds
from baselines import logger
import baselines.common.tf_util as tf_util
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from baselines.gail.trpo_mpi import traj_segment_generator, add_vtarg_and_adv, flatten_lists
from baselines.a2c.utils import make_path


class PPO1(BaseRLModel):
    def __init__(self, policy_fn, env, gamma=0.99, max_timesteps=0, timesteps_per_actorbatch=256, clip_param=0.2,
                 entcoeff=0.01, optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=64, lam=0.95, max_episodes=0,
                 max_iters=0, max_seconds=0, adam_epsilon=1e-5, schedule='linear', _init_setup_model=True):
        """
        Learning PPO with Stochastic Gradient Descent

        :param env: (Gym Environment) environment to train on
        :param policy_fn: (function (str, Gym Spaces, Gym Spaces): TensorFlow Tensor) creates the policy
        :param timesteps_per_actorbatch: (int) timesteps per actor per update
        :param clip_param: (float) clipping parameter epsilon
        :param entcoeff: (float) the entropy loss weight
        :param optim_epochs: (float) the optimizer's number of epochs
        :param optim_stepsize: (float) the optimizer's stepsize
        :param optim_batchsize: (int) the optimizer's the batch size
        :param gamma: (float) discount factor
        :param lam: (float) advantage estimation
        :param max_timesteps: (int) number of env steps to optimizer for
        :param max_episodes: (int) the maximum number of epochs
        :param max_iters: (int) the maximum number of iterations
        :param max_seconds: (int) the maximal duration
        :param adam_epsilon: (float) the epsilon value for the adam optimizer
        :param schedule: (str) The type of scheduler for the learning rate update ('linear', 'constant',
                                     'double_linear_con', 'middle_drop' or 'double_middle_drop')
        :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
        """
        super().__init__()

        self.sess = tf_util.single_threaded_session()

        self.policy_fn = policy_fn
        self.env = env
        self.ob_space = env.observation_space
        self.ac_space = env.action_space
        self.gamma = gamma
        self.max_timesteps = max_timesteps
        self.timesteps_per_actorbatch = timesteps_per_actorbatch
        self.clip_param = clip_param
        self.entcoeff = entcoeff
        self.optim_epochs = optim_epochs
        self.optim_stepsize = optim_stepsize
        self.optim_batchsize = optim_batchsize
        self.lam = lam
        self.max_episodes = max_episodes
        self.max_iters = max_iters
        self.max_seconds = max_seconds
        self.adam_epsilon = adam_epsilon
        self.schedule = schedule

        self.policy = None
        self.loss_names = None
        self.lossandgrad = None
        self.adam = None
        self.assign_old_eq_new = None
        self.compute_losses = None
        self.params = None

        self.setup_model()

    def setup_model(self):
        assert sum([self.max_iters > 0, self.max_timesteps > 0, self.max_episodes > 0,
                    self.max_seconds > 0]) == 1, "Only one time constraint permitted"

        # Construct network for new policy
        self.policy = policy = self.policy_fn("pi", self.ob_space, self.ac_space, sess=self.sess)

        # Network for old policy
        oldpi = self.policy_fn("oldpi", self.ob_space, self.ac_space, sess=self.sess,
                               placeholders={"obs": policy.obs_ph, "stochastic": policy.stochastic_ph})

        # Target advantage function (if applicable)
        atarg = tf.placeholder(dtype=tf.float32, shape=[None])

        # Empirical return
        ret = tf.placeholder(dtype=tf.float32, shape=[None])

        # learning rate multiplier, updated with schedule
        lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[])

        # Annealed cliping parameter epislon
        clip_param = self.clip_param * lrmult

        obs_ph = policy.obs_ph
        action_ph = policy.pdtype.sample_placeholder([None])

        kloldnew = oldpi.proba_distribution.kl(policy.proba_distribution)
        ent = policy.proba_distribution.entropy()
        meankl = tf.reduce_mean(kloldnew)
        meanent = tf.reduce_mean(ent)
        pol_entpen = (-self.entcoeff) * meanent

        # pnew / pold
        ratio = tf.exp(policy.proba_distribution.logp(action_ph) - oldpi.proba_distribution.logp(action_ph))

        # surrogate from conservative policy iteration
        surr1 = ratio * atarg
        surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg

        # PPO's pessimistic surrogate (L^CLIP)
        pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2))
        vf_loss = tf.reduce_mean(tf.square(policy.vpred - ret))
        total_loss = pol_surr + pol_entpen + vf_loss
        losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
        self.loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

        self.params = policy.get_trainable_variables()
        self.lossandgrad = tf_util.function([obs_ph, action_ph, atarg, ret, lrmult],
                                            losses + [tf_util.flatgrad(total_loss, self.params)])
        self.adam = MpiAdam(self.params, epsilon=self.adam_epsilon, sess=self.sess)

        self.assign_old_eq_new = tf_util.function(
            [], [], updates=[tf.assign(oldv, newv) for (oldv, newv) in
                             zipsame(oldpi.get_variables(), policy.get_variables())])
        self.compute_losses = tf_util.function([obs_ph, action_ph, atarg, ret, lrmult], losses)

        tf_util.initialize(sess=self.sess)

    def learn(self, callback=None, seed=None, log_interval=100):
        if seed is not None:
            set_global_seeds(seed)

        self.adam.sync()

        # Prepare for rollouts
        seg_gen = traj_segment_generator(self.policy, self.env, self.timesteps_per_actorbatch, stochastic=True)

        episodes_so_far = 0
        timesteps_so_far = 0
        iters_so_far = 0
        t_start = time.time()

        # rolling buffer for episode lengths
        lenbuffer = deque(maxlen=100)
        # rolling buffer for episode rewards
        rewbuffer = deque(maxlen=100)

        while True:
            if callback:
                callback(locals(), globals())
            if self.max_timesteps and timesteps_so_far >= self.max_timesteps:
                break
            elif self.max_episodes and episodes_so_far >= self.max_episodes:
                break
            elif self.max_iters and iters_so_far >= self.max_iters:
                break
            elif self.max_seconds and time.time() - t_start >= self.max_seconds:
                break

            if self.schedule == 'constant':
                cur_lrmult = 1.0
            elif self.schedule == 'linear':
                cur_lrmult = max(1.0 - float(timesteps_so_far) / self.max_timesteps, 0)
            else:
                raise NotImplementedError

            logger.log("********** Iteration %i ************" % iters_so_far)

            seg = seg_gen.__next__()
            add_vtarg_and_adv(seg, self.gamma, self.lam)

            # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
            obs_ph, action_ph, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]

            # predicted value function before udpate
            vpredbefore = seg["vpred"]

            # standardized advantage function estimate
            atarg = (atarg - atarg.mean()) / atarg.std()
            dataset = Dataset(dict(ob=obs_ph, ac=action_ph, atarg=atarg, vtarg=tdlamret),
                              shuffle=not self.policy.recurrent)
            optim_batchsize = self.optim_batchsize or obs_ph.shape[0]

            if hasattr(self.policy, "ob_rms"):
                # update running mean/std for policy
                self.policy.ob_rms.update(obs_ph)

            # set old parameter values to new parameter values
            self.assign_old_eq_new(sess=self.sess)
            logger.log("Optimizing...")
            logger.log(fmt_row(13, self.loss_names))

            # Here we do a bunch of optimization epochs over the data
            for _ in range(self.optim_epochs):
                # list of tuples, each of which gives the loss for a minibatch
                losses = []
                for batch in dataset.iterate_once(optim_batchsize):
                    *newlosses, grad = self.lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"],
                                                        cur_lrmult, sess=self.sess)
                    self.adam.update(grad, self.optim_stepsize * cur_lrmult)
                    losses.append(newlosses)
                logger.log(fmt_row(13, np.mean(losses, axis=0)))

            logger.log("Evaluating losses...")
            losses = []
            for batch in dataset.iterate_once(optim_batchsize):
                newlosses = self.compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult,
                                                sess=self.sess)
                losses.append(newlosses)
            mean_losses, _, _ = mpi_moments(losses, axis=0)
            logger.log(fmt_row(13, mean_losses))
            for (loss_val, name) in zipsame(mean_losses, self.loss_names):
                logger.record_tabular("loss_" + name, loss_val)
            logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))

            # local values
            lrlocal = (seg["ep_lens"], seg["ep_rets"])

            # list of tuples
            listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)
            lens, rews = map(flatten_lists, zip(*listoflrpairs))
            lenbuffer.extend(lens)
            rewbuffer.extend(rews)
            logger.record_tabular("EpLenMean", np.mean(lenbuffer))
            logger.record_tabular("EpRewMean", np.mean(rewbuffer))
            logger.record_tabular("EpThisIter", len(lens))
            episodes_so_far += len(lens)
            timesteps_so_far += sum(lens)
            iters_so_far += 1
            logger.record_tabular("EpisodesSoFar", episodes_so_far)
            logger.record_tabular("TimestepsSoFar", timesteps_so_far)
            logger.record_tabular("TimeElapsed", time.time() - t_start)
            if MPI.COMM_WORLD.Get_rank() == 0:
                logger.dump_tabular()

        return self

    def save(self, save_path):
        data = {
            "gamma": self.gamma,
            "timesteps_per_actorbatch": self.timesteps_per_actorbatch,
            "clip_param": self.clip_param,
            "entcoeff": self.entcoeff,
            "optim_epochs": self.optim_epochs,
            "optim_stepsize": self.optim_stepsize,
            "optim_batchsize": self.optim_batchsize,
            "lam": self.lam,
            "max_episodes": self.max_episodes,
            "max_iters": self.max_iters,
            "max_seconds": self.max_seconds,
            "adam_epsilon": self.adam_epsilon,
            "schedule": self.schedule,
            "ob_space": self.ob_space,
            "ac_space": self.ac_space
        }

        with open(".".join(save_path.split('.')[:-1]) + "_class.pkl", "wb") as file:
            cloudpickle.dump(data, file)

        parameters = self.sess.run(self.params)
        make_path(os.path.dirname(save_path))
        joblib.dump(parameters, save_path)

    @classmethod
    def load(cls, load_path, env, **kwargs):
        with open(".".join(load_path.split('.')[:-1]) + "_class.pkl", "rb") as file:
            data = cloudpickle.load(file)

        assert data["ob_space"] == env.observation_space, \
            "Error: the environment passed must have at least the same observation space as the model was trained on."
        assert data["ac_space"] == env.action_space, \
            "Error: the environment passed must have at least the same action space as the model was trained on."

        model = cls(policy_fn=data["policy_fn"], env=env, _init_setup_model=False)
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model.setup_model()

        loaded_params = joblib.load(load_path)
        restores = []
        for param, loaded_p in zip(model.params, loaded_params):
            restores.append(param.assign(loaded_p))
        model.sess.run(restores)

        return model
