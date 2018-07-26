from collections import deque
import time

import tensorflow as tf
import numpy as np
from mpi4py import MPI

from baselines.common import Dataset, explained_variance, fmt_row, zipsame, BaseRLModel
from baselines import logger
import baselines.common.tf_util as tf_util
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from baselines.gail.trpo_mpi import traj_segment_generator, add_vtarg_and_adv, flatten_lists


class PPO1(BaseRLModel):
    def __init__(self, policy, env, gamma=0.99, timesteps_per_actorbatch=256, clip_param=0.2, entcoeff=0.01,
                 optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=64, lam=0.95, adam_epsilon=1e-5,
                 schedule='linear', verbose=0, _init_setup_model=True):
        """
        Learning PPO with Stochastic Gradient Descent

        :param env: (Gym Environment) environment to train on
        :param policy: (function (str, Gym Spaces, Gym Spaces): TensorFlow Tensor) creates the policy
        :param timesteps_per_actorbatch: (int) timesteps per actor per update
        :param clip_param: (float) clipping parameter epsilon
        :param entcoeff: (float) the entropy loss weight
        :param optim_epochs: (float) the optimizer's number of epochs
        :param optim_stepsize: (float) the optimizer's stepsize
        :param optim_batchsize: (int) the optimizer's the batch size
        :param gamma: (float) discount factor
        :param lam: (float) advantage estimation
        :param adam_epsilon: (float) the epsilon value for the adam optimizer
        :param schedule: (str) The type of scheduler for the learning rate update ('linear', 'constant',
                                     'double_linear_con', 'middle_drop' or 'double_middle_drop')
        :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
        :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
        """
        super().__init__(env=env, requires_vec_env=False, verbose=verbose)

        self.policy = policy
        self.gamma = gamma
        self.timesteps_per_actorbatch = timesteps_per_actorbatch
        self.clip_param = clip_param
        self.entcoeff = entcoeff
        self.optim_epochs = optim_epochs
        self.optim_stepsize = optim_stepsize
        self.optim_batchsize = optim_batchsize
        self.lam = lam
        self.adam_epsilon = adam_epsilon
        self.schedule = schedule

        self.graph = None
        self.sess = None
        self.policy_pi = None
        self.loss_names = None
        self.lossandgrad = None
        self.adam = None
        self.assign_old_eq_new = None
        self.compute_losses = None
        self.params = None

        if _init_setup_model:
            self.setup_model()

    def setup_model(self):
        super().setup_model()

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf_util.single_threaded_session(graph=self.graph)

            # Construct network for new policy
            self.policy_pi = policy_pi = self.policy("pi", self.observation_space, self.action_space, sess=self.sess)

            # Network for old policy
            oldpi = self.policy("oldpi", self.observation_space, self.action_space, sess=self.sess,
                                placeholders={"obs": policy_pi.obs_ph, "processed_obs": policy_pi.processed_x,
                                              "stochastic": policy_pi.stochastic_ph})

            # Target advantage function (if applicable)
            atarg = tf.placeholder(dtype=tf.float32, shape=[None])

            # Empirical return
            ret = tf.placeholder(dtype=tf.float32, shape=[None])

            # learning rate multiplier, updated with schedule
            lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[])

            # Annealed cliping parameter epislon
            clip_param = self.clip_param * lrmult

            obs_ph = policy_pi.obs_ph
            action_ph = policy_pi.pdtype.sample_placeholder([None])

            kloldnew = oldpi.proba_distribution.kl(policy_pi.proba_distribution)
            ent = policy_pi.proba_distribution.entropy()
            meankl = tf.reduce_mean(kloldnew)
            meanent = tf.reduce_mean(ent)
            pol_entpen = (-self.entcoeff) * meanent

            # pnew / pold
            ratio = tf.exp(policy_pi.proba_distribution.logp(action_ph) - oldpi.proba_distribution.logp(action_ph))

            # surrogate from conservative policy iteration
            surr1 = ratio * atarg
            surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg

            # PPO's pessimistic surrogate (L^CLIP)
            pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2))
            vf_loss = tf.reduce_mean(tf.square(policy_pi.vpred - ret))
            total_loss = pol_surr + pol_entpen + vf_loss
            losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
            self.loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

            self.params = policy_pi.get_trainable_variables()
            self.lossandgrad = tf_util.function([obs_ph, action_ph, atarg, ret, lrmult],
                                                losses + [tf_util.flatgrad(total_loss, self.params)])
            self.adam = MpiAdam(self.params, epsilon=self.adam_epsilon, sess=self.sess)

            self.assign_old_eq_new = tf_util.function(
                [], [], updates=[tf.assign(oldv, newv) for (oldv, newv) in
                                 zipsame(oldpi.get_variables(), policy_pi.get_variables())])
            self.compute_losses = tf_util.function([obs_ph, action_ph, atarg, ret, lrmult], losses)

            tf_util.initialize(sess=self.sess)

    def learn(self, total_timesteps, callback=None, seed=None, log_interval=100):
        self._setup_learn(seed)

        with self.sess.as_default():
            self.adam.sync()

            # Prepare for rollouts
            seg_gen = traj_segment_generator(self.policy_pi, self.env, self.timesteps_per_actorbatch, stochastic=True)

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
                if total_timesteps and timesteps_so_far >= total_timesteps:
                    break

                if self.schedule == 'constant':
                    cur_lrmult = 1.0
                elif self.schedule == 'linear':
                    cur_lrmult = max(1.0 - float(timesteps_so_far) / total_timesteps, 0)
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
                                  shuffle=not self.policy_pi.recurrent)
                optim_batchsize = self.optim_batchsize or obs_ph.shape[0]

                if hasattr(self.policy_pi, "ob_rms"):
                    # update running mean/std for policy
                    self.policy_pi.ob_rms.update(obs_ph)

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
                timesteps_so_far += seg["total_timestep"]
                iters_so_far += 1
                logger.record_tabular("EpisodesSoFar", episodes_so_far)
                logger.record_tabular("TimestepsSoFar", timesteps_so_far)
                logger.record_tabular("TimeElapsed", time.time() - t_start)
                if self.verbose >= 1 and MPI.COMM_WORLD.Get_rank() == 0:
                    logger.dump_tabular()

        return self

    def predict(self, observation, state=None, mask=None):
        observation = np.array(observation).reshape(self.observation_space.shape)

        action, _ = self.policy_pi.act(True, observation)
        if self._vectorize_action:
            return [action], [None]
        else:
            return action, None

    def action_probability(self, observation, state=None, mask=None):
        observation = np.array(observation).reshape(self.observation_space.shape)

        neglogp0 = self.policy_pi.proba_distribution.neglogp(self.policy_pi.proba_distribution.sample())
        if self._vectorize_action:
            return [self._softmax(self.sess.run(neglogp0, feed_dict={self.policy_pi.obs_ph: observation}))]
        else:
            return self._softmax(self.sess.run(neglogp0, feed_dict={self.policy_pi.obs_ph: observation}))

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
            "adam_epsilon": self.adam_epsilon,
            "schedule": self.schedule,
            "verbose": self.verbose,
            "policy": self.policy,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_envs": self.n_envs
        }

        params = self.sess.run(self.params)

        self._save_to_file(save_path, data=data, params=params)

    @classmethod
    def load(cls, load_path, env=None, **kwargs):
        data, params = cls._load_from_file(load_path)

        model = cls(policy=data["policy"], env=None, _init_setup_model=False)
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model.set_env(env)
        model.setup_model()

        restores = []
        for param, loaded_p in zip(model.params, params):
            restores.append(param.assign(loaded_p))
        model.sess.run(restores)

        return model
