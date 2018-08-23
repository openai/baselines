import time
import numpy as np
from mpi4py import MPI
import tensorflow as tf
from utils.misc import zipsame
from policies.agent import Agent
from utils.console import colorize
from contextlib import contextmanager
from optimizers.mpi.mpi_adam import MpiAdam
from utils.tf_primitives import GetFlat, SetFromFlat


class TRPO(Agent):
    """
    TRPO: Algorithm
    """

    def __init__(
            self,
            policy,
            observation_space,
            action_space,
            entropy_coeff=0.00
    ):
        super(TRPO, self).__init__(name='TRPO')

        self.reset_graph_and_vars()
        self.entropy_coeff = entropy_coeff
        self.pi = policy("pi", observation_space, action_space)
        self.oldpi = policy("oldpi", observation_space, action_space)
        self.target_advantage = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
        self.empirical_return = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return

        self.observation = self.get_placeholder_cached(name="ob")
        self.action = self.pi.pdtype.sample_placeholder([None])
        self.loss
        self.optimize

    @Agent.define_scope
    def loss(self):
        kloldnew = self.oldpi.pd.kl(self.pi.pd)
        ent = self.pi.pd.entropy()
        meankl = tf.reduce_mean(kloldnew)
        meanent = tf.reduce_mean(ent)
        entbonus = self.entropy_coeff * meanent

        self.vferr = tf.reduce_mean(tf.square(self.pi.vpred - self.empirical_return))

        ratio = tf.exp(self.pi.pd.logp(self.action) - self.oldpi.pd.logp(self.action)) # advantage * pnew / pold
        surrgain = tf.reduce_mean(ratio * self.target_advantage)

        self.optimgain = surrgain + entbonus
        self.losses = [self.optimgain, meankl, entbonus, surrgain, meanent]
        self.loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

        return meankl

    @Agent.define_scope
    def optimize(self):
        self.nworkers = MPI.COMM_WORLD.Get_size()
        self.rank = MPI.COMM_WORLD.Get_rank()
        all_var_list = self.pi.get_trainable_variables()
        var_list = [v for v in all_var_list if
                    v.name.split("/")[1].startswith("pol")]

        vf_var_list = [v for v in all_var_list if
                       v.name.split("/")[1].startswith("vf")]
        self.vfadam = MpiAdam(vf_var_list)

        self.get_flat = GetFlat(var_list)
        self.set_from_flat = SetFromFlat(var_list)
        klgrads = tf.gradients(self.loss, var_list)
        flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None],
                                      name="flat_tan")
        shapes = [var.get_shape().as_list() for var in var_list]
        start = 0
        tangents = []
        for shape in shapes:
            sz = self.intprod(shape)
            tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
            start += sz

        gvp = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in
                        zipsame(klgrads, tangents)])

        fvp = self.flatten_gradients(gvp, var_list)

        self.assign_old_eq_new = self.function(
            [], [], updates=[
                tf.assign(oldv, newv) for (oldv, newv) in zipsame(
                    self.oldpi.get_variables(), self.pi.get_variables()
                )
            ])
        self.compute_losses = self.function(
            [self.observation, self.action, self.target_advantage],
            self.losses
        )

        self.compute_lossandgrad = self.function(
            [self.observation, self.action, self.target_advantage],
            self.losses + [self.flatten_gradients(self.optimgain, var_list)]
        )
        self.compute_fvp = self.function(
            [flat_tangent,
             self.observation,
             self.action,
             self.target_advantage],
            fvp
        )

        self.compute_vflossandgrad = self.function(
            [self.observation, self.empirical_return],
            self.flatten_gradients(self.vferr, vf_var_list)
        )

        # th_init = self.get_flat()
        # MPI.COMM_WORLD.Bcast(th_init, root=0)
        # self.set_from_flat(th_init)
        # self.vfadam.sync()
        # print("Init param sum", th_init.sum(), flush=True)

        # return get_flat, set_from_flat

    @contextmanager
    def timed(self, msg):
        if self.rank == 0:
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds" % (time.time() - tstart),
                           color='magenta'))
        else:
            yield

    def allmean(self, x):
        assert isinstance(x, np.ndarray)
        out = np.empty_like(x)
        MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
        out /= self.nworkers
        return out

    def traj_segment_generator(self, pi, env, horizon, stochastic):
        # Initialize state variables
        t = 0
        ac = env.action_space.sample()
        new = True
        rew = 0.0
        ob = env.reset()

        cur_ep_ret = 0
        cur_ep_len = 0
        ep_rets = []
        ep_lens = []

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
                    "ob": obs,
                    "rew": rews,
                    "vpred": vpreds,
                    "new": news,
                    "ac": acs,
                    "prevac": prevacs,
                    "nextvpred": vpred * (1 - new),
                    "ep_rets": ep_rets,
                    "ep_lens": ep_lens
                }
                _, vpred = pi.act(stochastic, ob)
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
        new = np.append(
            seg["new"], 0
        )  # last element is only used for last vtarg,
        # but we already zeroed it if last new = 1
        vpred = np.append(seg["vpred"], seg["nextvpred"])
        T = len(seg["rew"])
        seg["adv"] = gaelam = np.empty(T, 'float32')
        rew = seg["rew"]
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1 - new[t + 1]
            delta = rew[t] + gamma * vpred[t + 1] * nonterminal - vpred[t]
            gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        seg["tdlamret"] = seg["adv"] + seg["vpred"]

    def flatten_lists(self, listoflists):
        return [el for list_ in listoflists for el in list_]
