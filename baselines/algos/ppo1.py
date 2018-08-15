import time
import numpy as np
import tensorflow as tf
from mpi4py import MPI
from utils import logger
from collections import deque
from utils.misc import zipsame
from policies.model import Model
# from policies.tf_primitives import TfUtil
from mpi.mpi_adam import MpiAdam
from utils.dataset import Dataset
from utils.console import fmt_row
from mpi.mpi_moments import mpi_moments
from utils.math_util import explained_variance


class PPOSGD(Model):
    def __init__(
            self,
            env,
            policy,
            observation_space,
            action_space,
            timesteps_per_actorbatch,  # timesteps per actor per update
            clip_param,
            entcoeff,  # clipping parameter epsilon, entropy coeff
            optim_epochs,
            optim_stepsize,
            optim_batchsize,  # optimization hypers
            gamma,
            lam,  # advantage estimation
            max_timesteps=0,
            max_episodes=0,
            max_iters=0,
            max_seconds=0,  # time constraint
            callback=None,  # you can do anything in the callback, since it takes locals(), globals()
            adam_epsilon=1e-5,
            schedule='constant'  # annealing for stepsize parameters (epsilon and adam)
    ):
        self.env = env
        self.timesteps_per_actorbatch = timesteps_per_actorbatch
        self.clip_param = clip_param
        self.optim_epochs = optim_epochs
        self.optim_stepsize = optim_stepsize
        self.optim_batchsize = optim_batchsize
        self.gamma = gamma
        self.lam = lam
        self.max_timesteps = max_timesteps
        self.max_episodes = max_episodes
        self.max_iters = max_iters
        self.max_seconds = max_seconds
        self.callback = callback
        self.entcoeff = entcoeff
        self.adam_epsilon = adam_epsilon
        self.schedule = schedule
        super(PPOSGD, self).__init__(name='PPOSGD')

        # Construct network for new policy
        self.pi = policy("pi", observation_space, action_space)

        # Network for old policy
        self.oldpi = policy("oldpi", observation_space, action_space)

        # Target advantage function (if applicable)
        self.target_advantage = tf.placeholder(dtype=tf.float32, shape=[None])

        # Empirical return
        self.empirical_return = tf.placeholder(dtype=tf.float32, shape=[None])

        # learning rate multiplier, updated with schedule
        self.lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[])
        self.clip_param = self.clip_param * self.lrmult  # Annealed cliping parameter epislon

        self.observations = self.pi.get_placeholder_cached(name="ob")
        self.actions = self.pi.pdtype.sample_placeholder([None])
        self.loss
        self.rollouts

    @Model.define_scope
    def loss(self):
        kloldnew = self.oldpi.pd.kl(self.pi.pd)
        ent = self.pi.pd.entropy()
        meankl = tf.reduce_mean(kloldnew)
        meanent = tf.reduce_mean(ent)
        pol_entpen = (-self.entcoeff) * meanent

        ratio = tf.exp(
            self.pi.pd.logp(self.actions) - self.oldpi.pd.logp(self.actions)
        ) # pnew / pold
        surr1 = ratio * self.target_advantage  # surrogate from conservative policy iteration
        surr2 = tf.clip_by_value(
            ratio,
            1.0 - self.clip_param,
            1.0 + self.clip_param
        ) * self.target_advantage  #
        pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
        vf_loss = tf.reduce_mean(tf.square(self.pi.vpred - self.empirical_return))
        total_loss = pol_surr + pol_entpen + vf_loss
        losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
        self.loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

        var_list = self.pi.trainable_vars
        self.lossandgrad = self.function(
            inputs=[self.observations,
                    self.actions,
                    self.target_advantage,
                    self.empirical_return,
                    self.lrmult],
            outputs=losses + [self.flatten_gradients(total_loss, var_list)]
        )
        # adam = MpiAdam(var_list, epsilon=self.adam_epsilon)

        # assign_old_eq_new = self.function(
        #     [], [], updates=[tf.assign(oldv, newv) for (oldv, newv)
        #                      in zipsame(self.oldpi.variables, self.pi.variables)]
        # )
        self.adam = MpiAdam(self.pi.trainable_vars, epsilon=self.adam_epsilon)
        self.assign_old_eq_new = self.function(
            [], [],
            updates=[
                tf.assign(oldv, newv) for (oldv, newv) in
                zipsame(self.oldpi.variables, self.pi.variables)
            ]
        )
        compute_losses = self.function(
            inputs=[self.observations,
                    self.actions,
                    self.target_advantage,
                    self.empirical_return,
                    self.lrmult],
            outputs=losses
        )

        self.init_vars()
        self.adam.sync()
        return compute_losses

    # # @Model.define_scope
    # # def optimizer(self):
    # #     self.adam = MpiAdam(self.pi.trainable_vars, epsilon=self.adam_epsilon)
    # #     self.assign_old_eq_new = self.function(
    # #         [], [],
    # #         updates=[
    # #             tf.assign(oldv, newv) for (oldv, newv) in
    # #             zipsame(self.oldpi.variables, self.pi.variables)
    # #         ]
    # #     )
    # #     self.adam.sync()

    @Model.define_scope
    def rollouts(self):
        # Prepare for rollouts
        # ----------------------------------------
        seg_gen = self.traj_segment_generator(
            self.pi, self.env, self.timesteps_per_actorbatch, stochastic=True
        )

        episodes_so_far = 0
        timesteps_so_far = 0
        iters_so_far = 0
        tstart = time.time()
        lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
        rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards

        assert sum(
            [self.max_iters > 0,
             self.max_timesteps > 0,
             self.max_episodes > 0,
             self.max_seconds > 0]
        ) == 1, "Only one time constraint permitted"

        while True:
            if self.callback:
                self.callback(locals(), globals())
            if self.max_timesteps and timesteps_so_far >= self.max_timesteps:
                break
            elif self.max_episodes and episodes_so_far >= self.max_episodes:
                break
            elif self.max_iters and iters_so_far >= self.max_iters:
                break
            elif self.max_seconds and time.time() - tstart >= self.max_seconds:
                break

            if self.schedule == 'constant':
                cur_lrmult = 1.0
            elif self.schedule == 'linear':
                cur_lrmult =  max(
                    1.0 - float(timesteps_so_far) / self.max_timesteps, 0
                )
            else:
                raise NotImplementedError

            logger.log("********** Iteration %i ************" % iters_so_far)

            seg = seg_gen.__next__()
            self.add_vtarg_and_adv(seg, self.gamma, self.lam)

            ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], \
                seg["tdlamret"]
            vpredbefore = seg["vpred"] # predicted value function before udpate
            atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
            d = Dataset(
                dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret),
                shuffle=not self.pi.recurrent
            )
            optim_batchsize = self.optim_batchsize or ob.shape[0]

            if hasattr(self.pi, "ob_rms"):
                self.pi.ob_rms.update(ob) # update running mean/std for policy

            self.assign_old_eq_new()  # set old parameter values to new parameter values
            logger.log("Optimizing...")
            logger.log(fmt_row(13, self.loss_names))
            # Here we do a bunch of optimization epochs over the data
            for _ in range(self.optim_epochs):
                losses = [] # list of tuples, each of which gives the loss for a minibatch
                for batch in d.iterate_once(optim_batchsize):
                    *newlosses, g = self.lossandgrad(
                        batch["ob"],
                        batch["ac"],
                        batch["atarg"],
                        batch["vtarg"],
                        cur_lrmult
                    )
                    self.adam.update(g, self.optim_stepsize * cur_lrmult)
                    losses.append(newlosses)
                logger.log(fmt_row(13, np.mean(losses, axis=0)))

            logger.log("Evaluating losses...")
            losses = []
            for batch in d.iterate_once(optim_batchsize):
                newlosses = self.loss(
                    batch["ob"],
                    batch["ac"],
                    batch["atarg"],
                    batch["vtarg"],
                    cur_lrmult
                )
                losses.append(newlosses)
            meanlosses, _, _ = mpi_moments(losses, axis=0)
            logger.log(fmt_row(13, meanlosses))
            for (lossval, name) in zipsame(meanlosses, self.loss_names):
                logger.record_tabular("loss_"+name, lossval)
            logger.record_tabular(
                "ev_tdlam_before",
                explained_variance(vpredbefore, tdlamret)
            )
            lrlocal = (seg["ep_lens"], seg["ep_rets"])  # local values
            listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
            lens, rews = map(self.flatten_lists, zip(*listoflrpairs))
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
            logger.record_tabular("TimeElapsed", time.time() - tstart)
            if MPI.COMM_WORLD.Get_rank() == 0:
                logger.dump_tabular()

        return self.pi

    def traj_segment_generator(self, pi, env, horizon, stochastic):
        t = 0
        ac = env.action_space.sample() # not used, just so we have the datatype
        new = True  # marks if we're on first timestep of an episode
        ob = env.reset()

        cur_ep_ret = 0  # return in current episode
        cur_ep_len = 0  # len of current episode
        ep_rets = []  # returns of completed episodes in this segment
        ep_lens = []  # lengths of ...

        # Initialize history arrays
        obs = np.array([ob for _ in range(horizon)])
        rews = np.zeros(horizon, 'float32')
        vpreds = np.zeros(horizon, 'float32')
        news = np.zeros(horizon, 'int32')
        acs = np.array([ac for _ in range(horizon)])
        prevacs = acs.copy()

        while True:
            prevac = ac
            ac, vpred = pi.act(stochastic, ob)
            # Slight weirdness here because we need value function at time T
            # before returning segment [0, T-1] so we get the correct
            # terminal value
            if t > 0 and t % horizon == 0:
                yield {
                    "ob" : obs,
                    "rew" : rews,
                    "vpred" : vpreds,
                    "new" : news,
                    "ac" : acs,
                    "prevac" : prevacs,
                    "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets,
                    "ep_lens" : ep_lens
                }
                # Be careful!!! if you change the downstream algorithm to aggregate
                # several of these batches, then be sure to do a deepcopy
                ep_rets = []
                ep_lens = []
            i = t % horizon
            obs[i] = ob
            vpreds[i] = vpred
            news[i] = new
            acs[i] = ac
            prevacs[i] = prevac

            ob, rew, new, _ = env.step(ac)
            rews[i] = rew

            cur_ep_ret += rew
            cur_ep_len += 1
            if new:
                ep_rets.append(cur_ep_ret)
                ep_lens.append(cur_ep_len)
                cur_ep_ret = 0
                cur_ep_len = 0
                ob = env.reset()
            t += 1

    def add_vtarg_and_adv(self, seg, gamma, lam):
        """
        Compute target value using TD(lambda) estimator, and advantage
        with GAE(lambda)
        """
        new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
        vpred = np.append(seg["vpred"], seg["nextvpred"])
        T = len(seg["rew"])
        seg["adv"] = gaelam = np.empty(T, 'float32')
        rew = seg["rew"]
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1-new[t+1]
            delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
            gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam

        seg["tdlamret"] = seg["adv"] + seg["vpred"]

    def flatten_lists(self, listoflists):
        return [el for list_ in listoflists for el in list_]
