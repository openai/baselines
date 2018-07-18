import os
import time
import joblib

import numpy as np
import tensorflow as tf

from baselines import logger
from baselines.common import set_global_seeds, explained_variance, tf_util
from baselines.common.runners import AbstractEnvRunner
from baselines.a2c.utils import discount_with_dones, Scheduler, make_path, find_trainable_variables, calc_entropy, mse


class Model(object):
    def __init__(self, policy, ob_space, ac_space, n_envs, n_steps,
                 ent_coef=0.01, vf_coef=0.25, max_grad_norm=0.5, learning_rate=7e-4,
                 alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lr_schedule='linear'):
        """
        The A2C (Advantage Actor Critic) model class, https://arxiv.org/abs/1602.01783

        :param policy: (A2CPolicy) The policy model to use (MLP, CNN, LSTM, ...)
        :param ob_space: (Gym Space) Observation space
        :param ac_space: (Gym Space) Action space
        :param n_envs: (int) The number of environments
        :param n_steps: (int) The number of steps to run for each environment
        :param ent_coef: (float) Entropy coefficient for the loss caculation
        :param vf_coef: (float) Value function coefficient for the loss calculation
        :param max_grad_norm: (float) The maximum value for the gradient clipping
        :param learning_rate: (float) The learning rate
        :param alpha: (float) RMS prop optimizer decay
        :param epsilon: (float) RMS prop optimizer epsilon
        :param total_timesteps: (int) The total number of samples
        :param lr_schedule: (str) The type of scheduler for the learning rate update ('linear', 'constant',
                                 'double_linear_con', 'middle_drop' or 'double_middle_drop')
        """

        sess = tf_util.make_session()
        n_batch = n_envs * n_steps

        actions_ph = tf.placeholder(tf.int32, [n_batch])
        advs_ph = tf.placeholder(tf.float32, [n_batch])
        rewards_ph = tf.placeholder(tf.float32, [n_batch])
        learning_rate_ph = tf.placeholder(tf.float32, [])

        step_model = policy(sess, ob_space, ac_space, n_envs, 1, reuse=False)
        train_model = policy(sess, ob_space, ac_space, n_envs * n_steps, n_steps, reuse=True)

        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.policy, labels=actions_ph)
        pg_loss = tf.reduce_mean(advs_ph * neglogpac)
        vf_loss = mse(tf.squeeze(train_model.value_fn), rewards_ph)
        entropy = tf.reduce_mean(calc_entropy(train_model.policy))
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        params = find_trainable_variables("model")
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=learning_rate_ph, decay=alpha, epsilon=epsilon)
        _train = trainer.apply_gradients(grads)

        learning_rate = Scheduler(initial_value=learning_rate, n_values=total_timesteps, schedule=lr_schedule)

        def train(obs, states, rewards, masks, actions, values):
            advs = rewards - values
            for _ in range(len(obs)):
                cur_lr = learning_rate.value()
            td_map = {train_model.obs_ph: obs, actions_ph: actions, advs_ph: advs,
                      rewards_ph: rewards, learning_rate_ph: cur_lr}
            if states is not None:
                td_map[train_model.states_ph] = states
                td_map[train_model.masks_ph] = masks
            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train],
                td_map
            )
            return policy_loss, value_loss, policy_entropy

        def save(save_path):
            parameters = sess.run(params)
            make_path(os.path.dirname(save_path))
            joblib.dump(parameters, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for param, loaded_p in zip(params, loaded_params):
                restores.append(param.assign(loaded_p))
            sess.run(restores)

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)


class Runner(AbstractEnvRunner):
    def __init__(self, env, model, n_steps=5, gamma=0.99):
        """
        A runner to learn the policy of an environment for a model

        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        :param gamma: (float) Discount factor
        """
        super(Runner, self).__init__(env=env, model=model, n_steps=n_steps)
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


def learn(policy, env, seed, n_steps=5, total_timesteps=int(80e6), vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5,
          learning_rate=7e-4, lr_schedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=100):
    """
    Return a trained A2C model.

    :param policy: (A2CPolicy) The policy model to use (MLP, CNN, LSTM, ...)
    :param env: (Gym environment) The environment to learn from
    :param seed: (int) The initial seed for training
    :param n_steps: (int) The number of steps to run for each environment
    :param total_timesteps: (int) The total number of samples
    :param vf_coef: (float) Value function coefficient for the loss calculation
    :param ent_coef: (float) Entropy coefficient for the loss caculation
    :param max_grad_norm: (float) The maximum value for the gradient clipping
    :param learning_rate: (float) The learning rate
    :param lr_schedule: (str) The type of scheduler for the learning rate update ('linear', 'constant',
                                 'double_linear_con', 'middle_drop' or 'double_middle_drop')
    :param epsilon: (float) RMS prop optimizer epsilon
    :param alpha: (float) RMS prop optimizer decay
    :param gamma: (float) Discount factor
    :param log_interval: (int) The number of timesteps before logging.
    :return: (Model) A2C model
    """
    set_global_seeds(seed)

    n_envs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, n_envs=n_envs, n_steps=n_steps, ent_coef=ent_coef,
                  vf_coef=vf_coef, max_grad_norm=max_grad_norm, learning_rate=learning_rate,
                  alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps,
                  lr_schedule=lr_schedule)
    runner = Runner(env, model, n_steps=n_steps, gamma=gamma)

    n_batch = n_envs * n_steps
    t_start = time.time()
    for update in range(1, total_timesteps // n_batch + 1):
        obs, states, rewards, masks, actions, values = runner.run()
        _, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)
        n_seconds = time.time() - t_start
        fps = int((update * n_batch) / n_seconds)
        if update % log_interval == 0 or update == 1:
            explained_var = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update * n_batch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(explained_var))
            logger.dump_tabular()
    env.close()
    return model
