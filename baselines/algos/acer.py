import time
import tensorflow as tf
from utils import logger
from policies.agent import Agent
from utils.stats import EpisodeStats
from utils.schedules import Scheduler


# remove last step
def strip(var, nenvs, nsteps, flat=False):
    variables = Agent().batch_to_seq(var, nenvs, nsteps + 1, flat)
    return Agent().seq_to_batch(variables[:-1], flat)


def q_retrace(rewards, dones, q_i, values, rho_i, nenvs, nsteps, gamma):
    """
    Calculates q_retrace targets

    :param rewards: Rewards
    :param dones: Dones
    :param q_i: Q values for actions taken
    :param values: V values
    :param rho_i: Importance weight for each action
    :return: Q_retrace values
    """
    # list of len steps, shape [nenvs]
    rho_bar = Agent().batch_to_seq(tf.minimum(1.0, rho_i), nenvs, nsteps, True)
    # list of len steps, shape [nenvs]
    rs = Agent().batch_to_seq(rewards, nenvs, nsteps, True)
    # list of len steps, shape [nenvs]
    ds = Agent().batch_to_seq(dones, nenvs, nsteps, True)
    q_is = Agent().batch_to_seq(q_i, nenvs, nsteps, True)
    vs = Agent().batch_to_seq(values, nenvs, nsteps + 1, True)
    v_final = vs[-1]
    qret = v_final
    qrets = []
    for i in range(nsteps - 1, -1, -1):
        Agent().check_shape(
            [qret, ds[i], rs[i], rho_bar[i], q_is[i], vs[i]],
            [[nenvs]] * 6
        )
        qret = rs[i] + gamma * qret * (1.0 - ds[i])
        qrets.append(qret)
        qret = (rho_bar[i] * (qret - q_is[i])) + vs[i]
    qrets = qrets[::-1]
    qret = Agent().seq_to_batch(qrets, flat=True)
    return qret


# For ACER with PPO clipping instead of trust region
# def clip(ratio, eps_clip):
#     # assume 0 <= eps_clip <= 1
#     return tf.minimum(1 + eps_clip, tf.maximum(1 - eps_clip, ratio))

class Acer(Agent):
    def __init__(
            self,
            policy,
            observation_space,
            action_space,
            nenvs,
            nsteps,
            nstack,
            ent_coef,
            q_coef,
            gamma,
            max_grad_norm,
            lr,
            rprop_alpha,
            rprop_epsilon,
            total_timesteps,
            lrschedule,
            c,
            trust_region,
            alpha,
            delta
    ):
        # super(Acer, self).__init__(name='Acer')
        self.sess = self.init_session()
        nbatch = nenvs * nsteps
        self.c = c
        self.lr = lr
        self.nenvs = nenvs
        self.gamma = gamma
        self.delta = delta
        self.epsilon = 1e-6
        self.nsteps = nsteps
        self.q_coef = q_coef
        self.ent_coef = ent_coef
        self.nact = action_space.n
        self.lrschedule = lrschedule
        self.rprop_alpha = rprop_alpha
        self.trust_region = trust_region
        self.rprop_epsilon = rprop_epsilon
        self.max_grad_norm = max_grad_norm
        self.total_timesteps = total_timesteps

        self.action = tf.placeholder(tf.int32, [nbatch], name='actions')
        self.done = tf.placeholder(tf.float32, [nbatch], name='dones')
        self.reward = tf.placeholder(tf.float32, [nbatch], name='rewards')
        self.mu = tf.placeholder(tf.float32, [nbatch, self.nact], name='mus')
        self.learning_rate = tf.placeholder(
            tf.float32, [], name='learning_rate'
        )

        self.step_model = policy(
            self.sess,
            observation_space,
            action_space,
            nenvs,
            1,
            nstack,
            reuse=False
        )
        self.train_model = policy(
            self.sess,
            observation_space,
            action_space,
            nenvs,
            nsteps + 1,
            nstack,
            reuse=True
        )

        self.params = tf.trainable_variables()
        print("Params {}".format(len(self.params)))
        for var in self.params:
            print(var)

        # create polyak averaged model
        ema = tf.train.ExponentialMovingAverage(alpha)
        self.ema_apply_op = ema.apply(self.params)

        def custom_getter(getter, *args, **kwargs):
            v = ema.average(getter(*args, **kwargs))
            print(v.name)
            return v

        with tf.variable_scope("", custom_getter=custom_getter, reuse=True):
            self.polyak_model = policy(
                self.sess,
                observation_space,
                action_space,
                nenvs,
                nsteps + 1,
                nstack,
                reuse=True
            )

        self.initial_state = self.step_model.initial_state
        self.loss
        self.optimizer
        tf.global_variables_initializer().run(session=self.sess)

    ################################################################
    # Loss function                                                #
    ################################################################
    @Agent.define_scope
    def loss(self):
        # Notation: (var) = batch variable, (var)s = seqeuence
        # variable, (var)_i = variable index by action at step i
        # shape is [nenvs * (nsteps + 1)]
        v = tf.reduce_sum(self.train_model.pi * self.train_model.q, axis=-1)

        # strip off last step
        f, f_pol, q = map(
            lambda var: strip(var, self.nenvs, self.nsteps),
            [self.train_model.pi, self.polyak_model.pi, self.train_model.q]
        )
        # Get pi and q values for actions taken
        f_i = self.get_by_index(f, self.action)
        q_i = self.get_by_index(q, self.action)

        # Compute ratios for importance truncation
        rho = f / (self.mu + self.epsilon)
        rho_i = self.get_by_index(rho, self.action)

        # Calculate Q_retrace targets
        qret = q_retrace(
            self.reward,
            self.done,
            q_i,
            v,
            rho_i,
            self.nenvs,
            self.nsteps,
            self.gamma
        )

        # Calculate losses
        # Entropy
        entropy = tf.reduce_mean(self.categorical_entropy_softmax(f))

        # Policy Graident loss, with truncated importance sampling &
        # bias correction
        v = strip(v, self.nenvs, self.nsteps, True)
        self.check_shape([qret, v, rho_i, f_i], [[self.nenvs * self.nsteps]] * 4)
        self.check_shape([rho, f, q], [[self.nenvs * self.nsteps, self.nact]] * 2)

        # Truncated importance sampling
        adv = qret - v
        logf = tf.log(f_i + self.epsilon)
        # [nenvs * nsteps]
        gain_f = logf * tf.stop_gradient(adv * tf.minimum(self.c, rho_i))
        loss_f = -tf.reduce_mean(gain_f)

        # Bias correction for the truncation
        adv_bc = (q - tf.reshape(v, [self.nenvs * self.nsteps, 1]))  # [nenvs * nsteps, nact]
        logf_bc = tf.log(f + self.epsilon)  # / (f_old + eps)
        self.check_shape(
            [adv_bc, logf_bc],
            [[self.nenvs * self.nsteps, self.nact]] * 2
        )
        gain_bc = tf.reduce_sum(
            logf_bc * tf.stop_gradient(
                adv_bc * self.activation('relu')(
                    1.0 - (self.c / (rho + self.epsilon))
                ) * f
            ),
            axis=1
        )  # IMP: This is sum, as expectation wrt f

        loss_bc = -tf.reduce_mean(gain_bc)
        loss_policy = loss_f + loss_bc

        # Value/Q function loss, and explained variance
        self.check_shape([qret, q_i], [[self.nenvs * self.nsteps]]*2)
        ev = self.q_explained_variance(
            tf.reshape(q_i, [self.nenvs, self.nsteps]),
            tf.reshape(qret, [self.nenvs, self.nsteps])
        )
        loss_q = tf.reduce_mean(tf.square(tf.stop_gradient(qret) - q_i)*0.5)

        # Net loss
        self.check_shape([loss_policy, loss_q, entropy], [[]] * 3)
        loss = loss_policy + self.q_coef * loss_q - self.ent_coef * entropy

        return f, f_pol, entropy, loss_f, loss_bc, loss_policy, ev, loss_q, loss

    ################################################################
    # Optimizer                                                    #
    ################################################################
    @Agent.define_scope
    def optimizer(self):
        f, f_pol, entropy, loss_f, loss_bc, loss_policy, ev, loss_q, loss \
            = self.loss
        if self.trust_region:
            g = tf.gradients(
                - (loss_policy - self.ent_coef * entropy) * self.nsteps *
                self.nenvs,
                f
            )  # [nenvs * nsteps, nact]
            # k = tf.gradients(KL(f_pol || f), f)
            k = - f_pol / (f + self.epsilon)  # [nenvs * nsteps, nact] # Directly computed gradient of KL divergence wrt f
            k_dot_g = tf.reduce_sum(k * g, axis=-1)
            adj = tf.maximum(
                0.0, (tf.reduce_sum(k * g, axis=-1) - self.delta) /
                (tf.reduce_sum(tf.square(k), axis=-1) + self.epsilon)
            )  # [nenvs * nsteps]

            # Calculate stats (before doing adjustment) for logging.
            avg_norm_k = self.avg_norm(k)
            avg_norm_g = self.avg_norm(g)
            avg_norm_k_dot_g = tf.reduce_mean(tf.abs(k_dot_g))
            avg_norm_adj = tf.reduce_mean(tf.abs(adj))

            g = g - tf.reshape(adj, [self.nenvs * self.nsteps, 1]) * k
            # These are turst region adjusted gradients wrt f ie
            # statistics of policy pi
            grads_f = -g / (self.nenvs * self.nsteps)
            grads_policy = tf.gradients(f, self.params, grads_f)
            grads_q = tf.gradients(loss_q * self.q_coef, self.params)
            grads = [self.gradient_add(g1, g2, param)
                     for (g1, g2, param) in zip(grads_policy, grads_q, self.params)]

            avg_norm_grads_f = self.avg_norm(grads_f) * (self.nsteps * self.nenvs)
            norm_grads_q = tf.global_norm(grads_q)
            norm_grads_policy = tf.global_norm(grads_policy)
        else:
            grads = tf.gradients(loss, self.params)

        if self.max_grad_norm is not None:
            grads, norm_grads = tf.clip_by_global_norm(grads, self.max_grad_norm)
        grads = list(zip(grads, self.params))
        trainer = tf.train.RMSPropOptimizer(
            learning_rate=self.learning_rate,
            decay=self.rprop_alpha,
            epsilon=self.rprop_epsilon
        )
        _opt_op = trainer.apply_gradients(grads)

        # so when you call _train, you first do the gradient step, then you apply ema
        with tf.control_dependencies([_opt_op]):
            _train = tf.group(self.ema_apply_op)

        lr = Scheduler(
            v=self.lr, nvalues=self.total_timesteps, schedule=self.lrschedule
        )

        # Ops/Summaries to run, and their names for logging
        run_ops = [_train, loss, loss_q, entropy, loss_policy, loss_f,
                   loss_bc, ev, norm_grads]
        names_ops = ['loss', 'loss_q', 'entropy', 'loss_policy', 'loss_f',
                     'loss_bc', 'explained_variance',
                     'norm_grads']
        if self.trust_region:
            run_ops = run_ops + [norm_grads_q, norm_grads_policy,
                                 avg_norm_grads_f, avg_norm_k, avg_norm_g,
                                 avg_norm_k_dot_g, avg_norm_adj]
            names_ops = names_ops + ['norm_grads_q', 'norm_grads_policy',
                                     'avg_norm_grads_f', 'avg_norm_k',
                                     'avg_norm_g',
                                     'avg_norm_k_dot_g', 'avg_norm_adj']
        return lr, run_ops, names_ops

    ################################################################
    # Inference                                                    #
    ################################################################
    def predict(self, obs, actions, rewards, dones, mus, states, masks, steps):
        lr, run_ops, names_ops = self.optimizer
        cur_lr = lr.value_steps(steps)
        td_map = {
            self.train_model.X: obs,
            self.polyak_model.X: obs,
            self.action: actions,
            self.reward: rewards,
            self.done: dones,
            self.mu: mus,
            self.learning_rate: cur_lr
        }
        if states != []:
            td_map[self.train_model.S] = states
            td_map[self.train_model.M] = masks
            td_map[self.polyak_model.S] = states
            td_map[self.polyak_model.M] = masks
        return names_ops, self.sess.run(run_ops, td_map)[1:]  # strip off _train


class AgentEnv(object):
    def __init__(self, runner, model, buffer, log_interval):
        self.env_runner = runner
        self.model = model
        self.buffer = buffer
        self.log_interval = log_interval
        self.tstart = None
        self.episode_stats = EpisodeStats(runner.nsteps, runner.nenv)
        self.steps = None

    def call(self, on_policy):
        env_runner, model, buffer, steps = self.env_runner, self.model, \
            self.buffer, self.steps
        if on_policy:
            enc_obs, obs, actions, rewards, mus, dones, masks = env_runner.run()
            self.episode_stats.feed(rewards, dones)
            if buffer is not None:
                buffer.put(enc_obs, actions, rewards, mus, dones, masks)
        else:
            # get obs, actions, rewards, mus, dones from buffer.
            obs, actions, rewards, mus, dones, masks = buffer.get()

        # reshape stuff correctly
        obs = obs.reshape(env_runner.batch_ob_shape)
        actions = actions.reshape([env_runner.nbatch])
        rewards = rewards.reshape([env_runner.nbatch])
        mus = mus.reshape([env_runner.nbatch, env_runner.nact])
        dones = dones.reshape([env_runner.nbatch])
        masks = masks.reshape([env_runner.batch_ob_shape[0]])

        names_ops, values_ops = model.predict(
            obs,
            actions,
            rewards,
            dones,
            mus,
            model.initial_state,
            masks,
            steps
        )

        if on_policy and (int(steps/env_runner.nbatch) % self.log_interval == 0):
            logger.record_tabular("total_timesteps", steps)
            logger.record_tabular("fps", int(steps/(time.time() - self.tstart)))
            # IMP: In EpisodicLife env, during training, we get
            # done=True at each loss of life, not just at the terminal
            # state.  Thus, this is mean until end of life, not end of
            # episode.  For true episode rewards, see the monitor
            # files in the log folder.
            logger.record_tabular(
                "mean_episode_length",
                self.episode_stats.mean_length()
            )
            logger.record_tabular(
                "mean_episode_reward",
                self.episode_stats.mean_reward()
            )
            for name, val in zip(names_ops, values_ops):
                logger.record_tabular(name, float(val))
            logger.dump_tabular()
