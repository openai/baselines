import time

import numpy as np
import tensorflow as tf

from stable_baselines import logger
from stable_baselines.common import explained_variance, tf_util, BaseRLModel, SetVerbosity
from stable_baselines.common.policies import LstmPolicy
from stable_baselines.common.runners import AbstractEnvRunner
from stable_baselines.a2c.utils import discount_with_dones, Scheduler, find_trainable_variables, mse


class A2C(BaseRLModel):
    """
    The A2C (Advantage Actor Critic) model class, https://arxiv.org/abs/1602.01783

    :param policy: (ActorCriticPolicy) The policy model to use (MLP, CNN, LSTM, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) Discount factor
    :param n_steps: (int) The number of steps to run for each environment
    :param vf_coef: (float) Value function coefficient for the loss calculation
    :param ent_coef: (float) Entropy coefficient for the loss caculation
    :param max_grad_norm: (float) The maximum value for the gradient clipping
    :param learning_rate: (float) The learning rate
    :param alpha: (float) RMS prop optimizer decay
    :param epsilon: (float) RMS prop optimizer epsilon
    :param lr_schedule: (str) The type of scheduler for the learning rate update ('linear', 'constant',
                              'double_linear_con', 'middle_drop' or 'double_middle_drop')
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
                              (used only for loading)
    """

    def __init__(self, policy, env, gamma=0.99, n_steps=5, vf_coef=0.25, ent_coef=0.01, max_grad_norm=0.5,
                 learning_rate=7e-4, alpha=0.99, epsilon=1e-5, lr_schedule='linear', verbose=0, _init_setup_model=True):
        super(A2C, self).__init__(policy=policy, env=env, requires_vec_env=True, verbose=verbose)

        self.n_steps = n_steps
        self.gamma = gamma
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.alpha = alpha
        self.epsilon = epsilon
        self.lr_schedule = lr_schedule
        self.learning_rate = learning_rate

        self.graph = None
        self.sess = None
        self.learning_rate_ph = None
        self.n_batch = None
        self.actions_ph = None
        self.advs_ph = None
        self.rewards_ph = None
        self.pg_loss = None
        self.vf_loss = None
        self.entropy = None
        self.params = None
        self.apply_backprop = None
        self.train_model = None
        self.step_model = None
        self.step = None
        self.proba_step = None
        self.value = None
        self.initial_state = None
        self.learning_rate_schedule = None

        # if we are loading, it is possible the environment is not known, however the obs and action space are known
        if _init_setup_model:
            self.setup_model()

    def setup_model(self):
        with SetVerbosity(self.verbose):

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.sess = tf_util.make_session(graph=self.graph)

                self.n_batch = self.n_envs * self.n_steps

                n_batch_step = None
                n_batch_train = None
                if issubclass(self.policy, LstmPolicy):
                    n_batch_step = self.n_envs
                    n_batch_train = self.n_envs * self.n_steps

                step_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                         n_batch_step, reuse=False)
                train_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs,
                                          self.n_steps, n_batch_train, reuse=True)

                self.actions_ph = train_model.pdtype.sample_placeholder([None])
                self.advs_ph = tf.placeholder(tf.float32, [None])
                self.rewards_ph = tf.placeholder(tf.float32, [None])
                self.learning_rate_ph = tf.placeholder(tf.float32, [])

                neglogpac = train_model.proba_distribution.neglogp(self.actions_ph)
                self.entropy = tf.reduce_mean(train_model.proba_distribution.entropy())
                self.pg_loss = tf.reduce_mean(self.advs_ph * neglogpac)
                self.vf_loss = mse(tf.squeeze(train_model.value_fn), self.rewards_ph)
                loss = self.pg_loss - self.entropy * self.ent_coef + self.vf_loss * self.vf_coef

                self.params = find_trainable_variables("model")
                grads = tf.gradients(loss, self.params)
                if self.max_grad_norm is not None:
                    grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
                grads = list(zip(grads, self.params))
                trainer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate_ph, decay=self.alpha,
                                                    epsilon=self.epsilon)
                self.apply_backprop = trainer.apply_gradients(grads)

                self.train_model = train_model
                self.step_model = step_model
                self.step = step_model.step
                self.proba_step = step_model.proba_step
                self.value = step_model.value
                self.initial_state = step_model.initial_state
                tf.global_variables_initializer().run(session=self.sess)

    def _train_step(self, obs, states, rewards, masks, actions, values):
        """
        applies a training step to the model

        :param obs: ([float]) The input observations
        :param states: ([float]) The states (used for reccurent policies)
        :param rewards: ([float]) The rewards from the environment
        :param masks: ([bool]) Whether or not the episode is over (used for reccurent policies)
        :param actions: ([float]) The actions taken
        :param values: ([float]) The logits values
        :return: (float, float, float) policy loss, value loss, policy entropy
        """
        advs = rewards - values
        cur_lr = None
        for _ in range(len(obs)):
            cur_lr = self.learning_rate_schedule.value()
        assert cur_lr is not None, "Error: the observation input array cannon be empty"

        td_map = {self.train_model.obs_ph: obs, self.actions_ph: actions, self.advs_ph: advs,
                  self.rewards_ph: rewards, self.learning_rate_ph: cur_lr}
        if states is not None:
            td_map[self.train_model.states_ph] = states
            td_map[self.train_model.masks_ph] = masks

        policy_loss, value_loss, policy_entropy, _ = self.sess.run(
            [self.pg_loss, self.vf_loss, self.entropy, self.apply_backprop], td_map)
        return policy_loss, value_loss, policy_entropy

    def learn(self, total_timesteps, callback=None, seed=None, log_interval=100):
        with SetVerbosity(self.verbose):
            self._setup_learn(seed)

            self.learning_rate_schedule = Scheduler(initial_value=self.learning_rate, n_values=total_timesteps,
                                                    schedule=self.lr_schedule)

            runner = A2CRunner(self.env, self, n_steps=self.n_steps, gamma=self.gamma)

            t_start = time.time()
            for update in range(1, total_timesteps // self.n_batch + 1):
                obs, states, rewards, masks, actions, values = runner.run()
                _, value_loss, policy_entropy = self._train_step(obs, states, rewards, masks, actions, values)
                n_seconds = time.time() - t_start
                fps = int((update * self.n_batch) / n_seconds)

                if callback is not None:
                    callback(locals(), globals())

                if self.verbose >= 1 and (update % log_interval == 0 or update == 1):
                    explained_var = explained_variance(values, rewards)
                    logger.record_tabular("nupdates", update)
                    logger.record_tabular("total_timesteps", update * self.n_batch)
                    logger.record_tabular("fps", fps)
                    logger.record_tabular("policy_entropy", float(policy_entropy))
                    logger.record_tabular("value_loss", float(value_loss))
                    logger.record_tabular("explained_variance", float(explained_var))
                    logger.dump_tabular()

        return self

    def predict(self, observation, state=None, mask=None):
        if state is None:
            state = self.initial_state
        if mask is None:
            mask = [False for _ in range(self.n_envs)]
        observation = np.array(observation).reshape((-1,) + self.observation_space.shape)

        actions, _, states, _ = self.step(observation, state, mask)
        return actions, states

    def action_probability(self, observation, state=None, mask=None):
        if state is None:
            state = self.initial_state
        if mask is None:
            mask = [False for _ in range(self.n_envs)]
        observation = np.array(observation).reshape((-1,) + self.observation_space.shape)

        return self.proba_step(observation, state, mask)

    def save(self, save_path):
        data = {
            "gamma": self.gamma,
            "n_steps": self.n_steps,
            "vf_coef": self.vf_coef,
            "ent_coef": self.ent_coef,
            "max_grad_norm": self.max_grad_norm,
            "learning_rate": self.learning_rate,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
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


class A2CRunner(AbstractEnvRunner):
    def __init__(self, env, model, n_steps=5, gamma=0.99):
        """
        A runner to learn the policy of an environment for an a2c model

        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        :param gamma: (float) Discount factor
        """
        super(A2CRunner, self).__init__(env=env, model=model, n_steps=n_steps)
        self.gamma = gamma

    def run(self):
        """
        Run a learning step of the model

        :return: ([float], [float], [float], [bool], [float], [float])
                 observations, states, rewards, masks, actions, values
        """
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
        mb_states = self.states
        for _ in range(self.n_steps):
            actions, values, states, _ = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs, rewards, dones, _ = self.env.step(actions)
            self.states = states
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n] * 0
            self.obs = obs
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(0, 1)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(0, 1)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(0, 1)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(0, 1)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = self.model.value(self.obs, self.states, self.dones).tolist()
        # discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards

        # convert from [n_env, n_steps, ...] to [n_steps * n_env, ...]
        mb_rewards = mb_rewards.reshape(-1, *mb_rewards.shape[2:])
        mb_actions = mb_actions.reshape(-1, *mb_actions.shape[2:])
        mb_values = mb_values.reshape(-1, *mb_values.shape[2:])
        mb_masks = mb_masks.reshape(-1, *mb_masks.shape[2:])
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values
