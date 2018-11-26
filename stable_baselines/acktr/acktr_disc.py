"""
Discrete acktr
"""

import time

import tensorflow as tf
import numpy as np
from gym.spaces import Box

from stable_baselines import logger
from stable_baselines.common import explained_variance, ActorCriticRLModel, tf_util, SetVerbosity, TensorboardWriter
from stable_baselines.a2c.a2c import A2CRunner
from stable_baselines.a2c.utils import Scheduler, find_trainable_variables, calc_entropy, mse, \
    total_episode_reward_logger
from stable_baselines.acktr import kfac
from stable_baselines.common.policies import LstmPolicy, ActorCriticPolicy


class ACKTR(ActorCriticRLModel):
    """
    The ACKTR (Actor Critic using Kronecker-Factored Trust Region) model class, https://arxiv.org/abs/1708.05144

    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) Discount factor
    :param nprocs: (int) The number of threads for TensorFlow operations
    :param n_steps: (int) The number of steps to run for each environment
    :param ent_coef: (float) The weight for the entropic loss
    :param vf_coef: (float) The weight for the loss on the value function
    :param vf_fisher_coef: (float) The weight for the fisher loss on the value function
    :param learning_rate: (float) The initial learning rate for the RMS prop optimizer
    :param max_grad_norm: (float) The clipping value for the maximum gradient
    :param kfac_clip: (float) gradient clipping for Kullback leiber
    :param lr_schedule: (str) The type of scheduler for the learning rate update ('linear', 'constant',
                        'double_linear_con', 'middle_drop' or 'double_middle_drop')
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param async_eigen_decomp: (bool) Use async eigen decomposition
    """

    def __init__(self, policy, env, gamma=0.99, nprocs=1, n_steps=20, ent_coef=0.01, vf_coef=0.25, vf_fisher_coef=1.0,
                 learning_rate=0.25, max_grad_norm=0.5, kfac_clip=0.001, lr_schedule='linear', verbose=0,
                 tensorboard_log=None, _init_setup_model=True, async_eigen_decomp=False):

        super(ACKTR, self).__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=True,
                                    _init_setup_model=_init_setup_model)

        self.n_steps = n_steps
        self.gamma = gamma
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.vf_fisher_coef = vf_fisher_coef
        self.kfac_clip = kfac_clip
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate
        self.lr_schedule = lr_schedule
        self.nprocs = nprocs
        self.tensorboard_log = tensorboard_log
        self.async_eigen_decomp = async_eigen_decomp

        self.graph = None
        self.sess = None
        self.action_ph = None
        self.advs_ph = None
        self.rewards_ph = None
        self.pg_lr_ph = None
        self.model = None
        self.model2 = None
        self.logits = None
        self.entropy = None
        self.pg_loss = None
        self.vf_loss = None
        self.pg_fisher = None
        self.vf_fisher = None
        self.joint_fisher = None
        self.params = None
        self.grads_check = None
        self.optim = None
        self.train_op = None
        self.q_runner = None
        self.learning_rate_schedule = None
        self.train_model = None
        self.step_model = None
        self.step = None
        self.proba_step = None
        self.value = None
        self.initial_state = None
        self.n_batch = None
        self.summary = None
        self.episode_reward = None
        self.trained = False

        if _init_setup_model:
            self.setup_model()

    def setup_model(self):
        with SetVerbosity(self.verbose):

            assert issubclass(self.policy, ActorCriticPolicy), "Error: the input policy for the ACKTR model must be " \
                                                               "an instance of common.policies.ActorCriticPolicy."

            if isinstance(self.action_space, Box):
                raise NotImplementedError("WIP: ACKTR does not support Continuous actions yet.")

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.sess = tf_util.make_session(num_cpu=self.nprocs, graph=self.graph)

                n_batch_step = None
                n_batch_train = None
                if issubclass(self.policy, LstmPolicy):
                    n_batch_step = self.n_envs
                    n_batch_train = self.n_envs * self.n_steps

                self.model = step_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs,
                                                      1, n_batch_step, reuse=False)

                self.params = params = find_trainable_variables("model")

                with tf.variable_scope("train_model", reuse=True,
                                       custom_getter=tf_util.outer_scope_getter("train_model")):
                    self.model2 = train_model = self.policy(self.sess, self.observation_space, self.action_space,
                                                            self.n_envs, self.n_steps, n_batch_train,
                                                            reuse=True)

                with tf.variable_scope("loss", reuse=False, custom_getter=tf_util.outer_scope_getter("loss")):
                    self.advs_ph = advs_ph = tf.placeholder(tf.float32, [None])
                    self.rewards_ph = rewards_ph = tf.placeholder(tf.float32, [None])
                    self.pg_lr_ph = pg_lr_ph = tf.placeholder(tf.float32, [])
                    self.action_ph = action_ph = train_model.pdtype.sample_placeholder([None])

                    logpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.policy, labels=action_ph)
                    self.logits = train_model.policy

                    # training loss
                    pg_loss = tf.reduce_mean(advs_ph * logpac)
                    self.entropy = entropy = tf.reduce_mean(calc_entropy(train_model.policy))
                    self.pg_loss = pg_loss = pg_loss - self.ent_coef * entropy
                    self.vf_loss = vf_loss = mse(tf.squeeze(train_model.value_fn), rewards_ph)
                    train_loss = pg_loss + self.vf_coef * vf_loss

                    # Fisher loss construction
                    self.pg_fisher = pg_fisher_loss = -tf.reduce_mean(logpac)
                    sample_net = train_model.value_fn + tf.random_normal(tf.shape(train_model.value_fn))
                    self.vf_fisher = vf_fisher_loss = - self.vf_fisher_coef * tf.reduce_mean(
                        tf.pow(train_model.value_fn - tf.stop_gradient(sample_net), 2))
                    self.joint_fisher = pg_fisher_loss + vf_fisher_loss

                    tf.summary.scalar('entropy_loss', self.entropy)
                    tf.summary.scalar('policy_gradient_loss', pg_loss)
                    tf.summary.scalar('policy_gradient_fisher_loss', pg_fisher_loss)
                    tf.summary.scalar('value_function_loss', self.vf_loss)
                    tf.summary.scalar('value_function_fisher_loss', vf_fisher_loss)
                    tf.summary.scalar('loss', train_loss)

                    self.grads_check = tf.gradients(train_loss, params)

                with tf.variable_scope("input_info", reuse=False):
                    tf.summary.scalar('discounted_rewards', tf.reduce_mean(self.rewards_ph))
                    tf.summary.histogram('discounted_rewards', self.rewards_ph)
                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.pg_lr_ph))
                    tf.summary.histogram('learning_rate', self.pg_lr_ph)
                    tf.summary.scalar('advantage', tf.reduce_mean(self.advs_ph))
                    tf.summary.histogram('advantage', self.advs_ph)
                    if len(self.observation_space.shape) == 3:
                        tf.summary.image('observation', train_model.obs_ph)
                    else:
                        tf.summary.histogram('observation', train_model.obs_ph)

                with tf.variable_scope("kfac", reuse=False, custom_getter=tf_util.outer_scope_getter("kfac")):
                    with tf.device('/gpu:0'):
                        self.optim = optim = kfac.KfacOptimizer(learning_rate=pg_lr_ph, clip_kl=self.kfac_clip,
                                                                momentum=0.9, kfac_update=1,
                                                                epsilon=0.01, stats_decay=0.99,
                                                                async_eigen_decomp=self.async_eigen_decomp,
                                                                cold_iter=10,
                                                                max_grad_norm=self.max_grad_norm, verbose=self.verbose)

                        optim.compute_and_apply_stats(self.joint_fisher, var_list=params)

                self.train_model = train_model
                self.step_model = step_model
                self.step = step_model.step
                self.proba_step = step_model.proba_step
                self.value = step_model.value
                self.initial_state = step_model.initial_state
                tf.global_variables_initializer().run(session=self.sess)

                self.summary = tf.summary.merge_all()

    def _train_step(self, obs, states, rewards, masks, actions, values, update, writer):
        """
        applies a training step to the model

        :param obs: ([float]) The input observations
        :param states: ([float]) The states (used for recurrent policies)
        :param rewards: ([float]) The rewards from the environment
        :param masks: ([bool]) Whether or not the episode is over (used for recurrent policies)
        :param actions: ([float]) The actions taken
        :param values: ([float]) The logits values
        :param update: (int) the current step iteration
        :param writer: (TensorFlow Summary.writer) the writer for tensorboard
        :return: (float, float, float) policy loss, value loss, policy entropy
        """
        advs = rewards - values
        cur_lr = None
        for _ in range(len(obs)):
            cur_lr = self.learning_rate_schedule.value()
        assert cur_lr is not None, "Error: the observation input array cannon be empty"

        td_map = {self.train_model.obs_ph: obs, self.action_ph: actions, self.advs_ph: advs, self.rewards_ph: rewards,
                  self.pg_lr_ph: cur_lr}
        if states is not None:
            td_map[self.train_model.states_ph] = states
            td_map[self.train_model.masks_ph] = masks

        if writer is not None:
            # run loss backprop with summary, but once every 10 runs save the metadata (memory, compute time, ...)
            if (1 + update) % 10 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, policy_loss, value_loss, policy_entropy, _ = self.sess.run(
                    [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.train_op],
                    td_map, options=run_options, run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, 'step%d' % (update * (self.n_batch + 1)))
            else:
                summary, policy_loss, value_loss, policy_entropy, _ = self.sess.run(
                    [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.train_op], td_map)
            writer.add_summary(summary, update * (self.n_batch + 1))
        else:
            policy_loss, value_loss, policy_entropy, _ = self.sess.run(
                [self.pg_loss, self.vf_loss, self.entropy, self.train_op], td_map)

        return policy_loss, value_loss, policy_entropy

    def learn(self, total_timesteps, callback=None, seed=None, log_interval=100, tb_log_name="ACKTR"):
        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name) as writer:
            self._setup_learn(seed)
            self.n_batch = self.n_envs * self.n_steps

            self.learning_rate_schedule = Scheduler(initial_value=self.learning_rate, n_values=total_timesteps,
                                                    schedule=self.lr_schedule)

            # FIFO queue of the q_runner thread is closed at the end of the learn function.
            # As a result, it needs to be redefinied at every call
            with self.graph.as_default():
                with tf.variable_scope("kfac_apply", reuse=self.trained,
                                       custom_getter=tf_util.outer_scope_getter("kfac_apply")):
                    # Some of the variables are not in a scope when they are create
                    # so we make a note of any previously uninitialized variables
                    tf_vars = tf.global_variables()
                    is_uninitialized = self.sess.run([tf.is_variable_initialized(var) for var in tf_vars])
                    old_uninitialized_vars = [v for (v, f) in zip(tf_vars, is_uninitialized) if not f]

                    self.train_op, self.q_runner = self.optim.apply_gradients(list(zip(self.grads_check, self.params)))

                    # then we check for new uninitialized variables and initialize them
                    tf_vars = tf.global_variables()
                    is_uninitialized = self.sess.run([tf.is_variable_initialized(var) for var in tf_vars])
                    new_uninitialized_vars = [v for (v, f) in zip(tf_vars, is_uninitialized)
                                              if not f and v not in old_uninitialized_vars]

                    if len(new_uninitialized_vars) != 0:
                        self.sess.run(tf.variables_initializer(new_uninitialized_vars))

            self.trained = True

            runner = A2CRunner(self.env, self, n_steps=self.n_steps, gamma=self.gamma)
            self.episode_reward = np.zeros((self.n_envs,))

            t_start = time.time()
            coord = tf.train.Coordinator()
            if self.q_runner is not None:
                enqueue_threads = self.q_runner.create_threads(self.sess, coord=coord, start=True)
            else:
                enqueue_threads = []

            for update in range(1, total_timesteps // self.n_batch + 1):
                # true_reward is the reward without discount
                obs, states, rewards, masks, actions, values, true_reward = runner.run()
                policy_loss, value_loss, policy_entropy = self._train_step(obs, states, rewards, masks, actions, values,
                                                                           update, writer)
                n_seconds = time.time() - t_start
                fps = int((update * self.n_batch) / n_seconds)

                if writer is not None:
                    self.episode_reward = total_episode_reward_logger(self.episode_reward,
                                                                      true_reward.reshape((self.n_envs, self.n_steps)),
                                                                      masks.reshape((self.n_envs, self.n_steps)),
                                                                      writer, update * (self.n_batch + 1))

                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) == False:
                        break

                if self.verbose >= 1 and (update % log_interval == 0 or update == 1):
                    explained_var = explained_variance(values, rewards)
                    logger.record_tabular("nupdates", update)
                    logger.record_tabular("total_timesteps", update * self.n_batch)
                    logger.record_tabular("fps", fps)
                    logger.record_tabular("policy_entropy", float(policy_entropy))
                    logger.record_tabular("policy_loss", float(policy_loss))
                    logger.record_tabular("value_loss", float(value_loss))
                    logger.record_tabular("explained_variance", float(explained_var))
                    logger.dump_tabular()

            coord.request_stop()
            coord.join(enqueue_threads)

        return self

    def save(self, save_path):
        data = {
            "gamma": self.gamma,
            "nprocs": self.nprocs,
            "n_steps": self.n_steps,
            "vf_coef": self.vf_coef,
            "ent_coef": self.ent_coef,
            "vf_fisher_coef": self.vf_fisher_coef,
            "max_grad_norm": self.max_grad_norm,
            "learning_rate": self.learning_rate,
            "kfac_clip": self.kfac_clip,
            "lr_schedule": self.lr_schedule,
            "verbose": self.verbose,
            "policy": self.policy,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_envs": self.n_envs,
            "_vectorize_action": self._vectorize_action
        }

        params = self.sess.run(self.params)

        self._save_to_file(save_path, data=data, params=params)
