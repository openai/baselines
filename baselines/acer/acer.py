import time
import functools
import numpy as np
import tensorflow as tf
from baselines import logger

from baselines.common import set_global_seeds
from baselines.common.policies import build_policy
from baselines.common.tf_util import get_session, save_variables
from baselines.common.vec_env.vec_frame_stack import VecFrameStack

from baselines.a2c.utils import batch_to_seq, seq_to_batch
from baselines.a2c.utils import cat_entropy_softmax
from baselines.a2c.utils import Scheduler, find_trainable_variables
from baselines.a2c.utils import EpisodeStats
from baselines.a2c.utils import get_by_index, check_shape, avg_norm, gradient_add, q_explained_variance
from baselines.acer.buffer import Buffer
from baselines.acer.runner import Runner

# remove last step
def strip(var, nenvs, nsteps, flat = False):
    vars = batch_to_seq(var, nenvs, nsteps + 1, flat)
    return seq_to_batch(vars[:-1], flat)

def q_retrace(R, D, q_i, v, rho_i, nenvs, nsteps, gamma):
    """
    Calculates q_retrace targets

    :param R: Rewards
    :param D: Dones
    :param q_i: Q values for actions taken
    :param v: V values
    :param rho_i: Importance weight for each action
    :return: Q_retrace values
    """
    rho_bar = batch_to_seq(tf.minimum(1.0, rho_i), nenvs, nsteps, True)  # list of len steps, shape [nenvs]
    rs = batch_to_seq(R, nenvs, nsteps, True)  # list of len steps, shape [nenvs]
    ds = batch_to_seq(D, nenvs, nsteps, True)  # list of len steps, shape [nenvs]
    q_is = batch_to_seq(q_i, nenvs, nsteps, True)
    vs = batch_to_seq(v, nenvs, nsteps + 1, True)
    v_final = vs[-1]
    qret = v_final
    qrets = []
    for i in range(nsteps - 1, -1, -1):
        check_shape([qret, ds[i], rs[i], rho_bar[i], q_is[i], vs[i]], [[nenvs]] * 6)
        qret = rs[i] + gamma * qret * (1.0 - ds[i])
        qrets.append(qret)
        qret = (rho_bar[i] * (qret - q_is[i])) + vs[i]
    qrets = qrets[::-1]
    qret = seq_to_batch(qrets, flat=True)
    return qret

# For ACER with PPO clipping instead of trust region
# def clip(ratio, eps_clip):
#     # assume 0 <= eps_clip <= 1
#     return tf.minimum(1 + eps_clip, tf.maximum(1 - eps_clip, ratio))

class Model(object):
    def __init__(self, policy, ob_space, ac_space, nenvs, nsteps, ent_coef, q_coef, gamma, max_grad_norm, lr,
                 rprop_alpha, rprop_epsilon, total_timesteps, lrschedule,
                 c, trust_region, alpha, delta):

        sess = get_session()
        nact = ac_space.n
        nbatch = nenvs * nsteps

        A = tf.placeholder(tf.int32, [nbatch]) # actions
        D = tf.placeholder(tf.float32, [nbatch]) # dones
        R = tf.placeholder(tf.float32, [nbatch]) # rewards, not returns
        MU = tf.placeholder(tf.float32, [nbatch, nact]) # mu's
        LR = tf.placeholder(tf.float32, [])
        eps = 1e-6

        step_ob_placeholder = tf.placeholder(dtype=ob_space.dtype, shape=(nenvs,) + ob_space.shape)
        train_ob_placeholder = tf.placeholder(dtype=ob_space.dtype, shape=(nenvs*(nsteps+1),) + ob_space.shape)
        with tf.variable_scope('acer_model', reuse=tf.AUTO_REUSE):

            step_model = policy(observ_placeholder=step_ob_placeholder, sess=sess)
            train_model = policy(observ_placeholder=train_ob_placeholder, sess=sess)


        params = find_trainable_variables("acer_model")
        print("Params {}".format(len(params)))
        for var in params:
            print(var)

        # create polyak averaged model
        ema = tf.train.ExponentialMovingAverage(alpha)
        ema_apply_op = ema.apply(params)

        def custom_getter(getter, *args, **kwargs):
            v = ema.average(getter(*args, **kwargs))
            print(v.name)
            return v

        with tf.variable_scope("acer_model", custom_getter=custom_getter, reuse=True):
            polyak_model = policy(observ_placeholder=train_ob_placeholder, sess=sess)

        # Notation: (var) = batch variable, (var)s = seqeuence variable, (var)_i = variable index by action at step i

        # action probability distributions according to train_model, polyak_model and step_model
        # poilcy.pi is probability distribution parameters; to obtain distribution that sums to 1 need to take softmax
        train_model_p = tf.nn.softmax(train_model.pi)
        polyak_model_p = tf.nn.softmax(polyak_model.pi)
        step_model_p = tf.nn.softmax(step_model.pi)
        v = tf.reduce_sum(train_model_p * train_model.q, axis = -1) # shape is [nenvs * (nsteps + 1)]

        # strip off last step
        f, f_pol, q = map(lambda var: strip(var, nenvs, nsteps), [train_model_p, polyak_model_p, train_model.q])
        # Get pi and q values for actions taken
        f_i = get_by_index(f, A)
        q_i = get_by_index(q, A)

        # Compute ratios for importance truncation
        rho = f / (MU + eps)
        rho_i = get_by_index(rho, A)

        # Calculate Q_retrace targets
        qret = q_retrace(R, D, q_i, v, rho_i, nenvs, nsteps, gamma)

        # Calculate losses
        # Entropy
        # entropy = tf.reduce_mean(strip(train_model.pd.entropy(), nenvs, nsteps))
        entropy = tf.reduce_mean(cat_entropy_softmax(f))

        # Policy Graident loss, with truncated importance sampling & bias correction
        v = strip(v, nenvs, nsteps, True)
        check_shape([qret, v, rho_i, f_i], [[nenvs * nsteps]] * 4)
        check_shape([rho, f, q], [[nenvs * nsteps, nact]] * 2)

        # Truncated importance sampling
        adv = qret - v
        logf = tf.log(f_i + eps)
        gain_f = logf * tf.stop_gradient(adv * tf.minimum(c, rho_i))  # [nenvs * nsteps]
        loss_f = -tf.reduce_mean(gain_f)

        # Bias correction for the truncation
        adv_bc = (q - tf.reshape(v, [nenvs * nsteps, 1]))  # [nenvs * nsteps, nact]
        logf_bc = tf.log(f + eps) # / (f_old + eps)
        check_shape([adv_bc, logf_bc], [[nenvs * nsteps, nact]]*2)
        gain_bc = tf.reduce_sum(logf_bc * tf.stop_gradient(adv_bc * tf.nn.relu(1.0 - (c / (rho + eps))) * f), axis = 1) #IMP: This is sum, as expectation wrt f
        loss_bc= -tf.reduce_mean(gain_bc)

        loss_policy = loss_f + loss_bc

        # Value/Q function loss, and explained variance
        check_shape([qret, q_i], [[nenvs * nsteps]]*2)
        ev = q_explained_variance(tf.reshape(q_i, [nenvs, nsteps]), tf.reshape(qret, [nenvs, nsteps]))
        loss_q = tf.reduce_mean(tf.square(tf.stop_gradient(qret) - q_i)*0.5)

        # Net loss
        check_shape([loss_policy, loss_q, entropy], [[]] * 3)
        loss = loss_policy + q_coef * loss_q - ent_coef * entropy

        if trust_region:
            g = tf.gradients(- (loss_policy - ent_coef * entropy) * nsteps * nenvs, f) #[nenvs * nsteps, nact]
            # k = tf.gradients(KL(f_pol || f), f)
            k = - f_pol / (f + eps) #[nenvs * nsteps, nact] # Directly computed gradient of KL divergence wrt f
            k_dot_g = tf.reduce_sum(k * g, axis=-1)
            adj = tf.maximum(0.0, (tf.reduce_sum(k * g, axis=-1) - delta) / (tf.reduce_sum(tf.square(k), axis=-1) + eps)) #[nenvs * nsteps]

            # Calculate stats (before doing adjustment) for logging.
            avg_norm_k = avg_norm(k)
            avg_norm_g = avg_norm(g)
            avg_norm_k_dot_g = tf.reduce_mean(tf.abs(k_dot_g))
            avg_norm_adj = tf.reduce_mean(tf.abs(adj))

            g = g - tf.reshape(adj, [nenvs * nsteps, 1]) * k
            grads_f = -g/(nenvs*nsteps) # These are turst region adjusted gradients wrt f ie statistics of policy pi
            grads_policy = tf.gradients(f, params, grads_f)
            grads_q = tf.gradients(loss_q * q_coef, params)
            grads = [gradient_add(g1, g2, param) for (g1, g2, param) in zip(grads_policy, grads_q, params)]

            avg_norm_grads_f = avg_norm(grads_f) * (nsteps * nenvs)
            norm_grads_q = tf.global_norm(grads_q)
            norm_grads_policy = tf.global_norm(grads_policy)
        else:
            grads = tf.gradients(loss, params)

        if max_grad_norm is not None:
            grads, norm_grads = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=rprop_alpha, epsilon=rprop_epsilon)
        _opt_op = trainer.apply_gradients(grads)

        # so when you call _train, you first do the gradient step, then you apply ema
        with tf.control_dependencies([_opt_op]):
            _train = tf.group(ema_apply_op)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        # Ops/Summaries to run, and their names for logging
        run_ops = [_train, loss, loss_q, entropy, loss_policy, loss_f, loss_bc, ev, norm_grads]
        names_ops = ['loss', 'loss_q', 'entropy', 'loss_policy', 'loss_f', 'loss_bc', 'explained_variance',
                     'norm_grads']
        if trust_region:
            run_ops = run_ops + [norm_grads_q, norm_grads_policy, avg_norm_grads_f, avg_norm_k, avg_norm_g, avg_norm_k_dot_g,
                                 avg_norm_adj]
            names_ops = names_ops + ['norm_grads_q', 'norm_grads_policy', 'avg_norm_grads_f', 'avg_norm_k', 'avg_norm_g',
                                     'avg_norm_k_dot_g', 'avg_norm_adj']

        def train(obs, actions, rewards, dones, mus, states, masks, steps):
            cur_lr = lr.value_steps(steps)
            td_map = {train_model.X: obs, polyak_model.X: obs, A: actions, R: rewards, D: dones, MU: mus, LR: cur_lr}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
                td_map[polyak_model.S] = states
                td_map[polyak_model.M] = masks

            return names_ops, sess.run(run_ops, td_map)[1:]  # strip off _train

        def _step(observation, **kwargs):
            return step_model._evaluate([step_model.action, step_model_p, step_model.state], observation, **kwargs)



        self.train = train
        self.save = functools.partial(save_variables, sess=sess, variables=params)
        self.train_model = train_model
        self.step_model = step_model
        self._step = _step
        self.step = self.step_model.step

        self.initial_state = step_model.initial_state
        tf.global_variables_initializer().run(session=sess)


class Acer():
    def __init__(self, runner, model, buffer, log_interval):
        self.runner = runner
        self.model = model
        self.buffer = buffer
        self.log_interval = log_interval
        self.tstart = None
        self.episode_stats = EpisodeStats(runner.nsteps, runner.nenv)
        self.steps = None

    def call(self, on_policy):
        runner, model, buffer, steps = self.runner, self.model, self.buffer, self.steps
        if on_policy:
            enc_obs, obs, actions, rewards, mus, dones, masks = runner.run()
            self.episode_stats.feed(rewards, dones)
            if buffer is not None:
                buffer.put(enc_obs, actions, rewards, mus, dones, masks)
        else:
            # get obs, actions, rewards, mus, dones from buffer.
            obs, actions, rewards, mus, dones, masks = buffer.get()


        # reshape stuff correctly
        obs = obs.reshape(runner.batch_ob_shape)
        actions = actions.reshape([runner.nbatch])
        rewards = rewards.reshape([runner.nbatch])
        mus = mus.reshape([runner.nbatch, runner.nact])
        dones = dones.reshape([runner.nbatch])
        masks = masks.reshape([runner.batch_ob_shape[0]])

        names_ops, values_ops = model.train(obs, actions, rewards, dones, mus, model.initial_state, masks, steps)

        if on_policy and (int(steps/runner.nbatch) % self.log_interval == 0):
            logger.record_tabular("total_timesteps", steps)
            logger.record_tabular("fps", int(steps/(time.time() - self.tstart)))
            # IMP: In EpisodicLife env, during training, we get done=True at each loss of life, not just at the terminal state.
            # Thus, this is mean until end of life, not end of episode.
            # For true episode rewards, see the monitor files in the log folder.
            logger.record_tabular("mean_episode_length", self.episode_stats.mean_length())
            logger.record_tabular("mean_episode_reward", self.episode_stats.mean_reward())
            for name, val in zip(names_ops, values_ops):
                logger.record_tabular(name, float(val))
            logger.dump_tabular()


def learn(network, env, seed=None, nsteps=20, total_timesteps=int(80e6), q_coef=0.5, ent_coef=0.01,
          max_grad_norm=10, lr=7e-4, lrschedule='linear', rprop_epsilon=1e-5, rprop_alpha=0.99, gamma=0.99,
          log_interval=100, buffer_size=50000, replay_ratio=4, replay_start=10000, c=10.0,
          trust_region=True, alpha=0.99, delta=1, load_path=None, **network_kwargs):

    '''
    Main entrypoint for ACER (Actor-Critic with Experience Replay) algorithm (https://arxiv.org/pdf/1611.01224.pdf)
    Train an agent with given network architecture on a given environment using ACER.

    Parameters:
    ----------

    network:            policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                        specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                        tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                        neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                        See baselines.common/policies.py/lstm for more details on using recurrent nets in policies

    env:                environment. Needs to be vectorized for parallel environment simulation.
                        The environments produced by gym.make can be wrapped using baselines.common.vec_env.DummyVecEnv class.

    nsteps:             int, number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                        nenv is number of environment copies simulated in parallel) (default: 20)

    nstack:             int, size of the frame stack, i.e. number of the frames passed to the step model. Frames are stacked along channel dimension
                        (last image dimension) (default: 4)

    total_timesteps:    int, number of timesteps (i.e. number of actions taken in the environment) (default: 80M)

    q_coef:             float, value function loss coefficient in the optimization objective (analog of vf_coef for other actor-critic methods)

    ent_coef:           float, policy entropy coefficient in the optimization objective (default: 0.01)

    max_grad_norm:      float, gradient norm clipping coefficient. If set to None, no clipping. (default: 10),

    lr:                 float, learning rate for RMSProp (current implementation has RMSProp hardcoded in) (default: 7e-4)

    lrschedule:         schedule of learning rate. Can be 'linear', 'constant', or a function [0..1] -> [0..1] that takes fraction of the training progress as input and
                        returns fraction of the learning rate (specified as lr) as output

    rprop_epsilon:      float, RMSProp epsilon (stabilizes square root computation in denominator of RMSProp update) (default: 1e-5)

    rprop_alpha:        float, RMSProp decay parameter (default: 0.99)

    gamma:              float, reward discounting factor (default: 0.99)

    log_interval:       int, number of updates between logging events (default: 100)

    buffer_size:        int, size of the replay buffer (default: 50k)

    replay_ratio:       int, now many (on average) batches of data to sample from the replay buffer take after batch from the environment (default: 4)

    replay_start:       int, the sampling from the replay buffer does not start until replay buffer has at least that many samples (default: 10k)

    c:                  float, importance weight clipping factor (default: 10)

    trust_region        bool, whether or not algorithms estimates the gradient KL divergence between the old and updated policy and uses it to determine step size  (default: True)

    delta:              float, max KL divergence between the old policy and updated policy (default: 1)

    alpha:              float, momentum factor in the Polyak (exponential moving average) averaging of the model parameters (default: 0.99)

    load_path:          str, path to load the model from (default: None)

    **network_kwargs:               keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                                    For instance, 'mlp' network architecture has arguments num_hidden and num_layers.

    '''

    print("Running Acer Simple")
    print(locals())
    set_global_seeds(seed)
    if not isinstance(env, VecFrameStack):
        env = VecFrameStack(env, 1)

    policy = build_policy(env, network, estimate_q=True, **network_kwargs)
    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space

    nstack = env.nstack
    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps,
                  ent_coef=ent_coef, q_coef=q_coef, gamma=gamma,
                  max_grad_norm=max_grad_norm, lr=lr, rprop_alpha=rprop_alpha, rprop_epsilon=rprop_epsilon,
                  total_timesteps=total_timesteps, lrschedule=lrschedule, c=c,
                  trust_region=trust_region, alpha=alpha, delta=delta)

    runner = Runner(env=env, model=model, nsteps=nsteps)
    if replay_ratio > 0:
        buffer = Buffer(env=env, nsteps=nsteps, size=buffer_size)
    else:
        buffer = None
    nbatch = nenvs*nsteps
    acer = Acer(runner, model, buffer, log_interval)
    acer.tstart = time.time()

    for acer.steps in range(0, total_timesteps, nbatch): #nbatch samples, 1 on_policy call and multiple off-policy calls
        acer.call(on_policy=True)
        if replay_ratio > 0 and buffer.has_atleast(replay_start):
            n = np.random.poisson(replay_ratio)
            for _ in range(n):
                acer.call(on_policy=False)  # no simulation steps in this

    return model
