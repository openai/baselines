import os
import time
import joblib

import numpy as np
import tensorflow as tf

from baselines import logger
from baselines.common import set_global_seeds, explained_variance, tf_util, BaseRLModel
from baselines.common.runners import AbstractEnvRunner
from baselines.a2c.utils import discount_with_dones, Scheduler, make_path, find_trainable_variables, calc_entropy, mse


class A2C(BaseRLModel):
    def __init__(self, policy, env, gamma=0.99, n_steps=5, total_timesteps=int(80e6), vf_coef=0.25,
                 ent_coef=0.01, max_grad_norm=0.5, learning_rate=7e-4, alpha=0.99, epsilon=1e-5, lr_schedule='linear'):
        """
        The A2C (Advantage Actor Critic) model class, https://arxiv.org/abs/1602.01783

        :param policy: (A2CPolicy) The policy model to use (MLP, CNN, LSTM, ...)
        :param env: (Gym environment) The environment to learn from
        :param gamma: (float) Discount factor
        :param n_steps: (int) The number of steps to run for each environment
        :param total_timesteps: (int) The total number of samples
        :param vf_coef: (float) Value function coefficient for the loss calculation
        :param ent_coef: (float) Entropy coefficient for the loss caculation
        :param max_grad_norm: (float) The maximum value for the gradient clipping
        :param learning_rate: (float) The learning rate
        :param alpha: (float) RMS prop optimizer decay
        :param epsilon: (float) RMS prop optimizer epsilon
        :param lr_schedule: (str) The type of scheduler for the learning rate update ('linear', 'constant',
                                 'double_linear_con', 'middle_drop' or 'double_middle_drop')
        """
        super(A2C, self).__init__()
        sess = tf_util.make_session()
        n_envs = env.num_envs
        n_batch = n_envs * n_steps

        self.actions_ph = tf.placeholder(tf.int32, [n_batch])
        self.advs_ph = tf.placeholder(tf.float32, [n_batch])
        self.rewards_ph = tf.placeholder(tf.float32, [n_batch])
        self.learning_rate_ph = tf.placeholder(tf.float32, [])

        ob_space = env.observation_space
        ac_space = env.action_space

        step_model = policy(sess, ob_space, ac_space, n_envs, 1, reuse=False)
        train_model = policy(sess, ob_space, ac_space, n_envs * n_steps, n_steps, reuse=True)

        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.policy, labels=self.actions_ph)
        self.pg_loss = tf.reduce_mean(self.advs_ph * neglogpac)
        self.vf_loss = mse(tf.squeeze(train_model.value_fn), self.rewards_ph)
        self.entropy = tf.reduce_mean(calc_entropy(train_model.policy))
        loss = self.pg_loss - self.entropy * ent_coef + self.vf_loss * vf_coef

        self.params = find_trainable_variables("model")
        grads = tf.gradients(loss, self.params)
        if max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, self.params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate_ph, decay=alpha, epsilon=epsilon)
        self.apply_backprop = trainer.apply_gradients(grads)

        self.learning_rate = Scheduler(initial_value=learning_rate, n_values=total_timesteps, schedule=lr_schedule)

        self.env = env
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.n_batch = n_batch
        self.gamma = gamma
        self.total_timesteps = total_timesteps

        self.sess = sess
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        tf.global_variables_initializer().run(session=sess)

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
            cur_lr = self.learning_rate.value()
        assert cur_lr is not None, "Error: the observation input array cannon be empty"

        td_map = {self.train_model.obs_ph: obs, self.actions_ph: actions, self.advs_ph: advs,
                  self.rewards_ph: rewards, self.learning_rate_ph: cur_lr}
        if states is not None:
            td_map[self.train_model.states_ph] = states
            td_map[self.train_model.masks_ph] = masks

        policy_loss, value_loss, policy_entropy, _ = self.sess.run(
            [self.pg_loss, self.vf_loss, self.entropy, self.apply_backprop], td_map)
        return policy_loss, value_loss, policy_entropy

    def learn(self, callback=None, seed=None, log_interval=100):
        if seed is not None:
            set_global_seeds(seed)

        runner = A2CRunner(self.env, self, n_steps=self.n_steps, gamma=self.gamma)

        t_start = time.time()
        for update in range(1, self.total_timesteps // self.n_batch + 1):
            obs, states, rewards, masks, actions, values = runner.run()
            _, value_loss, policy_entropy = self._train_step(obs, states, rewards, masks, actions, values)
            n_seconds = time.time() - t_start
            fps = int((update * self.n_batch) / n_seconds)

            if callback is not None:
                callback(locals(), globals())

            if update % log_interval == 0 or update == 1:
                explained_var = explained_variance(values, rewards)
                logger.record_tabular("nupdates", update)
                logger.record_tabular("total_timesteps", update * self.n_batch)
                logger.record_tabular("fps", fps)
                logger.record_tabular("policy_entropy", float(policy_entropy))
                logger.record_tabular("value_loss", float(value_loss))
                logger.record_tabular("explained_variance", float(explained_var))
                logger.dump_tabular()

        return self

    def save(self, save_path):
        parameters = self.sess.run(self.params)
        make_path(os.path.dirname(save_path))
        joblib.dump(parameters, save_path)

    def load(self, load_path):
        loaded_params = joblib.load(load_path)
        restores = []
        for param, loaded_p in zip(self.params, loaded_params):
            restores.append(param.assign(loaded_p))
        self.sess.run(restores)


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
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
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
        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values
