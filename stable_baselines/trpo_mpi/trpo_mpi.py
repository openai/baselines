import time
from contextlib import contextmanager
from collections import deque

from mpi4py import MPI
import tensorflow as tf
import numpy as np

import stable_baselines.common.tf_util as tf_util
from stable_baselines.common import explained_variance, zipsame, dataset, fmt_row, colorize, ActorCriticRLModel, \
    SetVerbosity, TensorboardWriter
from stable_baselines import logger
from stable_baselines.common.mpi_adam import MpiAdam
from stable_baselines.common.cg import conjugate_gradient
from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.a2c.utils import find_trainable_variables, total_episode_reward_logger
from stable_baselines.trpo_mpi.utils import traj_segment_generator, add_vtarg_and_adv, flatten_lists
# from stable_baselines.gail.statistics import Stats


class TRPO(ActorCriticRLModel):
    def __init__(self, policy, env, gamma=0.99, timesteps_per_batch=1024, max_kl=0.01, cg_iters=10, lam=0.98,
                 entcoeff=0.0, cg_damping=1e-2, vf_stepsize=3e-4, vf_iters=3, verbose=0, tensorboard_log=None,
                 _init_setup_model=True):
        """
        learns a TRPO policy using the given environment

        :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
        :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
        :param gamma: (float) the discount value
        :param timesteps_per_batch: (int) the number of timesteps to run per batch (horizon)
        :param max_kl: (float) the kullback leiber loss threshold
        :param cg_iters: (int) the number of iterations for the conjugate gradient calculation
        :param lam: (float) GAE factor
        :param entcoeff: (float) the weight for the entropy loss
        :param cg_damping: (float) the compute gradient dampening factor
        :param vf_stepsize: (float) the value function stepsize
        :param vf_iters: (int) the value function's number iterations for learning
        :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
        :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
        :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
        """
        super(TRPO, self).__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=False,
                                   _init_setup_model=_init_setup_model)

        self.using_gail = False
        self.timesteps_per_batch = timesteps_per_batch
        self.cg_iters = cg_iters
        self.cg_damping = cg_damping
        self.gamma = gamma
        self.lam = lam
        self.max_kl = max_kl
        self.vf_iters = vf_iters
        self.vf_stepsize = vf_stepsize
        self.entcoeff = entcoeff
        self.tensorboard_log = tensorboard_log

        # GAIL Params
        self.pretrained_weight = None
        self.hidden_size_adversary = 100
        self.adversary_entcoeff = 1e-3
        self.expert_dataset = None
        self.save_per_iter = 1
        self.checkpoint_dir = "/tmp/gail/ckpt/"
        self.g_step = 1
        self.d_step = 1
        self.task_name = "task_name"
        self.d_stepsize = 3e-4

        self.graph = None
        self.sess = None
        self.policy_pi = None
        self.loss_names = None
        self.assign_old_eq_new = None
        self.compute_losses = None
        self.compute_lossandgrad = None
        self.compute_fvp = None
        self.compute_vflossandgrad = None
        self.d_adam = None
        self.vfadam = None
        self.get_flat = None
        self.set_from_flat = None
        self.timed = None
        self.allmean = None
        self.nworkers = None
        self.rank = None
        self.reward_giver = None
        self.step = None
        self.proba_step = None
        self.initial_state = None
        self.params = None
        self.summary = None
        self.episode_reward = None

        if _init_setup_model:
            self.setup_model()

    def setup_model(self):
        # prevent import loops
        from stable_baselines.gail.adversary import TransitionClassifier

        with SetVerbosity(self.verbose):

            assert issubclass(self.policy, ActorCriticPolicy), "Error: the input policy for the TRPO model must be " \
                                                               "an instance of common.policies.ActorCriticPolicy."

            self.nworkers = MPI.COMM_WORLD.Get_size()
            self.rank = MPI.COMM_WORLD.Get_rank()
            np.set_printoptions(precision=3)

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.sess = tf_util.single_threaded_session(graph=self.graph)

                if self.using_gail:
                    self.reward_giver = TransitionClassifier(self.env, self.hidden_size_adversary,
                                                             entcoeff=self.adversary_entcoeff)

                # Construct network for new policy
                self.policy_pi = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                             None, reuse=False)

                # Network for old policy
                with tf.variable_scope("oldpi", reuse=False):
                    old_policy = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                             None, reuse=False)

                with tf.variable_scope("loss", reuse=False):
                    atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
                    ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return

                    observation = self.policy_pi.obs_ph
                    action = self.policy_pi.pdtype.sample_placeholder([None])

                    kloldnew = old_policy.proba_distribution.kl(self.policy_pi.proba_distribution)
                    ent = self.policy_pi.proba_distribution.entropy()
                    meankl = tf.reduce_mean(kloldnew)
                    meanent = tf.reduce_mean(ent)
                    entbonus = self.entcoeff * meanent

                    vferr = tf.reduce_mean(tf.square(self.policy_pi.value_fn[:, 0] - ret))

                    # advantage * pnew / pold
                    ratio = tf.exp(self.policy_pi.proba_distribution.logp(action) -
                                   old_policy.proba_distribution.logp(action))
                    surrgain = tf.reduce_mean(ratio * atarg)

                    optimgain = surrgain + entbonus
                    losses = [optimgain, meankl, entbonus, surrgain, meanent]
                    self.loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

                    dist = meankl

                    all_var_list = tf_util.get_trainable_vars("model")
                    var_list = [v for v in all_var_list if "/vf" not in v.name and "/q/" not in v.name]
                    vf_var_list = [v for v in all_var_list if "/pi" not in v.name and "/logstd" not in v.name]

                    self.get_flat = tf_util.GetFlat(var_list, sess=self.sess)
                    self.set_from_flat = tf_util.SetFromFlat(var_list, sess=self.sess)

                    klgrads = tf.gradients(dist, var_list)
                    flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
                    shapes = [var.get_shape().as_list() for var in var_list]
                    start = 0
                    tangents = []
                    for shape in shapes:
                        var_size = tf_util.intprod(shape)
                        tangents.append(tf.reshape(flat_tangent[start: start + var_size], shape))
                        start += var_size
                    gvp = tf.add_n([tf.reduce_sum(grad * tangent)
                                    for (grad, tangent) in zipsame(klgrads, tangents)])  # pylint: disable=E1111
                    fvp = tf_util.flatgrad(gvp, var_list)

                    tf.summary.scalar('entropy_loss', meanent)
                    tf.summary.scalar('policy_gradient_loss', optimgain)
                    tf.summary.scalar('value_function_loss', surrgain)
                    tf.summary.scalar('approximate_kullback-leiber', meankl)
                    tf.summary.scalar('loss', optimgain + meankl + entbonus + surrgain + meanent)

                    self.assign_old_eq_new = \
                        tf_util.function([], [], updates=[tf.assign(oldv, newv) for (oldv, newv) in
                                                          zipsame(tf_util.get_globals_vars("oldpi"),
                                                          tf_util.get_globals_vars("model"))])
                    self.compute_losses = tf_util.function([observation, old_policy.obs_ph, action, atarg], losses)
                    self.compute_fvp = tf_util.function([flat_tangent, observation, old_policy.obs_ph, action, atarg],
                                                        fvp)
                    self.compute_vflossandgrad = tf_util.function([observation, old_policy.obs_ph, ret],
                                                                  tf_util.flatgrad(vferr, vf_var_list))

                    @contextmanager
                    def timed(msg):
                        if self.rank == 0 and self.verbose >= 1:
                            print(colorize(msg, color='magenta'))
                            start_time = time.time()
                            yield
                            print(colorize("done in %.3f seconds" % (time.time() - start_time), color='magenta'))
                        else:
                            yield

                    def allmean(arr):
                        assert isinstance(arr, np.ndarray)
                        out = np.empty_like(arr)
                        MPI.COMM_WORLD.Allreduce(arr, out, op=MPI.SUM)
                        out /= self.nworkers
                        return out

                    tf_util.initialize(sess=self.sess)

                    th_init = self.get_flat()
                    MPI.COMM_WORLD.Bcast(th_init, root=0)
                    self.set_from_flat(th_init)

                with tf.variable_scope("Adam_mpi", reuse=False):
                    self.vfadam = MpiAdam(vf_var_list, sess=self.sess)
                    if self.using_gail:
                        self.d_adam = MpiAdam(self.reward_giver.get_trainable_variables())
                        self.d_adam.sync()
                    self.vfadam.sync()

                with tf.variable_scope("input_info", reuse=False):
                    tf.summary.scalar('discounted_rewards', tf.reduce_mean(ret))
                    tf.summary.histogram('discounted_rewards', ret)
                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.vf_stepsize))
                    tf.summary.histogram('learning_rate', self.vf_stepsize)
                    tf.summary.scalar('advantage', tf.reduce_mean(atarg))
                    tf.summary.histogram('advantage', atarg)
                    tf.summary.scalar('kl_clip_range', tf.reduce_mean(self.max_kl))
                    tf.summary.histogram('kl_clip_range', self.max_kl)
                    if len(self.observation_space.shape) == 3:
                        tf.summary.image('observation', observation)
                    else:
                        tf.summary.histogram('observation', observation)

                self.timed = timed
                self.allmean = allmean

                self.step = self.policy_pi.step
                self.proba_step = self.policy_pi.proba_step
                self.initial_state = self.policy_pi.initial_state

                self.params = find_trainable_variables("model")
                if self.using_gail:
                    self.params.extend(self.reward_giver.get_trainable_variables())

                self.summary = tf.summary.merge_all()

                self.compute_lossandgrad = \
                    tf_util.function([observation, old_policy.obs_ph, action, atarg, ret],
                                     [self.summary, tf_util.flatgrad(optimgain, var_list)] + losses)

    def learn(self, total_timesteps, callback=None, seed=None, log_interval=100, tb_log_name="TRPO"):
        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name) as writer:
            self._setup_learn(seed)

            with self.sess.as_default():
                seg_gen = traj_segment_generator(self.policy_pi, self.env, self.timesteps_per_batch,
                                                 reward_giver=self.reward_giver, gail=self.using_gail)

                episodes_so_far = 0
                timesteps_so_far = 0
                iters_so_far = 0
                t_start = time.time()
                lenbuffer = deque(maxlen=40)  # rolling buffer for episode lengths
                rewbuffer = deque(maxlen=40)  # rolling buffer for episode rewards
                self.episode_reward = np.zeros((self.n_envs,))

                true_rewbuffer = None
                if self.using_gail:
                    true_rewbuffer = deque(maxlen=40)
                    #  Stats not used for now
                    #  g_loss_stats = Stats(loss_names)
                    #  d_loss_stats = Stats(reward_giver.loss_name)
                    #  ep_stats = Stats(["True_rewards", "Rewards", "Episode_length"])

                    # if provide pretrained weight
                    if self.pretrained_weight is not None:
                        tf_util.load_state(self.pretrained_weight, var_list=tf_util.get_globals_vars("pi"),
                                           sess=self.sess)

                while True:
                    if callback:
                        callback(locals(), globals())
                    if total_timesteps and timesteps_so_far >= total_timesteps:
                        break

                    logger.log("********** Iteration %i ************" % iters_so_far)

                    def fisher_vector_product(vec):
                        return self.allmean(self.compute_fvp(vec, *fvpargs, sess=self.sess)) + self.cg_damping * vec
                    # ------------------ Update G ------------------
                    logger.log("Optimizing Policy...")
                    # g_step = 1 when not using GAIL
                    mean_losses = None
                    vpredbefore = None
                    tdlamret = None
                    observation = None
                    action = None
                    seg = None
                    for k in range(self.g_step):
                        with self.timed("sampling"):
                            seg = seg_gen.__next__()
                        add_vtarg_and_adv(seg, self.gamma, self.lam)
                        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
                        observation, action, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
                        vpredbefore = seg["vpred"]  # predicted value function before udpate
                        atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate

                        # true_rew is the reward without discount
                        if writer is not None:
                            self.episode_reward = total_episode_reward_logger(self.episode_reward,
                                                                              seg["true_rew"].reshape(
                                                                                  (self.n_envs, -1)),
                                                                              seg["dones"].reshape((self.n_envs, -1)),
                                                                              writer, timesteps_so_far)

                        args = seg["ob"], seg["ob"], seg["ac"], atarg
                        fvpargs = [arr[::5] for arr in args]

                        self.assign_old_eq_new(sess=self.sess)

                        with self.timed("computegrad"):
                            steps = timesteps_so_far + (k + 1) * (seg["total_timestep"] / self.g_step)
                            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                            run_metadata = tf.RunMetadata()
                            # run loss backprop with summary, and save the metadata (memory, compute time, ...)
                            if writer is not None:
                                summary, grad, *lossbefore = self.compute_lossandgrad(*args, tdlamret, sess=self.sess,
                                                                                      options=run_options,
                                                                                      run_metadata=run_metadata)
                                writer.add_run_metadata(run_metadata, 'step%d' % steps)
                                writer.add_summary(summary, steps)
                            else:
                                _, grad, *lossbefore = self.compute_lossandgrad(*args, tdlamret, sess=self.sess,
                                                                                options=run_options,
                                                                                run_metadata=run_metadata)

                        lossbefore = self.allmean(np.array(lossbefore))
                        grad = self.allmean(grad)
                        if np.allclose(grad, 0):
                            logger.log("Got zero gradient. not updating")
                        else:
                            with self.timed("cg"):
                                stepdir = conjugate_gradient(fisher_vector_product, grad, cg_iters=self.cg_iters,
                                                             verbose=self.rank == 0 and self.verbose >= 1)
                            assert np.isfinite(stepdir).all()
                            shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
                            # abs(shs) to avoid taking square root of negative values
                            lagrange_multiplier = np.sqrt(abs(shs) / self.max_kl)
                            # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
                            fullstep = stepdir / lagrange_multiplier
                            expectedimprove = grad.dot(fullstep)
                            surrbefore = lossbefore[0]
                            stepsize = 1.0
                            thbefore = self.get_flat()
                            thnew = None
                            for _ in range(10):
                                thnew = thbefore + fullstep * stepsize
                                self.set_from_flat(thnew)
                                mean_losses = surr, kl_loss, *_ = self.allmean(
                                    np.array(self.compute_losses(*args, sess=self.sess)))
                                improve = surr - surrbefore
                                logger.log("Expected: %.3f Actual: %.3f" % (expectedimprove, improve))
                                if not np.isfinite(mean_losses).all():
                                    logger.log("Got non-finite value of losses -- bad!")
                                elif kl_loss > self.max_kl * 1.5:
                                    logger.log("violated KL constraint. shrinking step.")
                                elif improve < 0:
                                    logger.log("surrogate didn't improve. shrinking step.")
                                else:
                                    logger.log("Stepsize OK!")
                                    break
                                stepsize *= .5
                            else:
                                logger.log("couldn't compute a good step")
                                self.set_from_flat(thbefore)
                            if self.nworkers > 1 and iters_so_far % 20 == 0:
                                # list of tuples
                                paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), self.vfadam.getflat().sum()))
                                assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])

                        with self.timed("vf"):
                            for _ in range(self.vf_iters):
                                for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]),
                                                                         include_final_partial_batch=False,
                                                                         batch_size=128):
                                    grad = self.allmean(self.compute_vflossandgrad(mbob, mbob, mbret, sess=self.sess))
                                    self.vfadam.update(grad, self.vf_stepsize)

                    for (loss_name, loss_val) in zip(self.loss_names, mean_losses):
                        logger.record_tabular(loss_name, loss_val)

                    logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))

                    if self.using_gail:
                        # ------------------ Update D ------------------
                        logger.log("Optimizing Discriminator...")
                        logger.log(fmt_row(13, self.reward_giver.loss_name))
                        ob_expert, ac_expert = self.expert_dataset.get_next_batch(len(observation))
                        batch_size = len(observation) // self.d_step
                        d_losses = []  # list of tuples, each of which gives the loss for a minibatch
                        for ob_batch, ac_batch in dataset.iterbatches((observation, action),
                                                                      include_final_partial_batch=False,
                                                                      batch_size=batch_size):
                            ob_expert, ac_expert = self.expert_dataset.get_next_batch(len(ob_batch))
                            # update running mean/std for reward_giver
                            if hasattr(self.reward_giver, "obs_rms"):
                                self.reward_giver.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0))
                            *newlosses, grad = self.reward_giver.lossandgrad(ob_batch, ac_batch, ob_expert, ac_expert)
                            self.d_adam.update(self.allmean(grad), self.d_stepsize)
                            d_losses.append(newlosses)
                        logger.log(fmt_row(13, np.mean(d_losses, axis=0)))

                        lrlocal = (seg["ep_lens"], seg["ep_rets"], seg["ep_true_rets"])  # local values
                        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
                        lens, rews, true_rets = map(flatten_lists, zip(*listoflrpairs))
                        true_rewbuffer.extend(true_rets)
                    else:
                        lrlocal = (seg["ep_lens"], seg["ep_rets"])  # local values
                        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
                        lens, rews = map(flatten_lists, zip(*listoflrpairs))
                    lenbuffer.extend(lens)
                    rewbuffer.extend(rews)

                    logger.record_tabular("EpLenMean", np.mean(lenbuffer))
                    logger.record_tabular("EpRewMean", np.mean(rewbuffer))
                    if self.using_gail:
                        logger.record_tabular("EpTrueRewMean", np.mean(true_rewbuffer))
                    logger.record_tabular("EpThisIter", len(lens))
                    episodes_so_far += len(lens)
                    timesteps_so_far += seg["total_timestep"]
                    iters_so_far += 1

                    logger.record_tabular("EpisodesSoFar", episodes_so_far)
                    logger.record_tabular("TimestepsSoFar", timesteps_so_far)
                    logger.record_tabular("TimeElapsed", time.time() - t_start)

                    if self.verbose >= 1 and self.rank == 0:
                        logger.dump_tabular()

        return self

    def save(self, save_path):
        data = {
            "gamma": self.gamma,
            "timesteps_per_batch": self.timesteps_per_batch,
            "max_kl": self.max_kl,
            "cg_iters": self.cg_iters,
            "lam": self.lam,
            "entcoeff": self.entcoeff,
            "cg_damping": self.cg_damping,
            "vf_stepsize": self.vf_stepsize,
            "vf_iters": self.vf_iters,
            "pretrained_weight": self.pretrained_weight,
            "reward_giver": self.reward_giver,
            "expert_dataset": self.expert_dataset,
            "save_per_iter": self.save_per_iter,
            "checkpoint_dir": self.checkpoint_dir,
            "g_step": self.g_step,
            "d_step": self.d_step,
            "task_name": self.task_name,
            "d_stepsize": self.d_stepsize,
            "using_gail": self.using_gail,
            "verbose": self.verbose,
            "policy": self.policy,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_envs": self.n_envs,
            "_vectorize_action": self._vectorize_action
        }

        params = self.sess.run(self.params)

        self._save_to_file(save_path, data=data, params=params)
