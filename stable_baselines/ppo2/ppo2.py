import os
import time
import joblib
from collections import deque
import sys
import multiprocessing

import numpy as np
import tensorflow as tf

from stable_baselines import logger
from stable_baselines.common import explained_variance
from stable_baselines.common.runners import AbstractEnvRunner


class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, n_batch_act, n_batch_train, n_steps, ent_coef, vf_coef,
                 max_grad_norm):
        """
        The PPO (Proximal Policy Optimization) model class https://arxiv.org/abs/1707.06347.
        It shares policies with A2C.

        :param policy: (A2CPolicy) The policy model to use (MLP, CNN, LSTM, ...)
        :param ob_space: (Gym Spaces) Observation space
        :param ac_space: (Gym Spaces) Action space
        :param n_batch_act: (int) Minibatch size for the actor policy, used mostly for reccurent policies
        :param n_batch_train: (int) Minibatch size during training
        :param n_steps: (int) The number of steps to run for each environment
        :param ent_coef: (float) Entropy coefficient for the loss caculation
        :param vf_coef: (float) Value function coefficient for the loss calculation
        :param max_grad_norm: (float) The maximum value for the gradient clipping
        """

        n_cpu = multiprocessing.cpu_count()
        if sys.platform == 'darwin':
            n_cpu //= 2

        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=n_cpu,
                                inter_op_parallelism_threads=n_cpu)
        config.gpu_options.allow_growth = True  # pylint: disable=E1101

        sess = tf.Session(config=config)

        act_model = policy(sess, ob_space, ac_space, n_batch_act, 1, reuse=False)
        train_model = policy(sess, ob_space, ac_space, n_batch_train, n_steps, reuse=True)

        action_ph = train_model.pdtype.sample_placeholder([None])
        advs_ph = tf.placeholder(tf.float32, [None])
        rewards_ph = tf.placeholder(tf.float32, [None])
        old_neglog_pac_ph = tf.placeholder(tf.float32, [None])
        old_vpred_ph = tf.placeholder(tf.float32, [None])
        learning_rate_ph = tf.placeholder(tf.float32, [])
        clip_range_ph = tf.placeholder(tf.float32, [])

        neglogpac = train_model.proba_distribution.neglogp(action_ph)
        entropy = tf.reduce_mean(train_model.proba_distribution.entropy())

        vpred = train_model.value_fn
        vpredclipped = old_vpred_ph \
                       + tf.clip_by_value(train_model.value_fn - old_vpred_ph, - clip_range_ph, clip_range_ph)
        vf_losses1 = tf.square(vpred - rewards_ph)
        vf_losses2 = tf.square(vpredclipped - rewards_ph)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        ratio = tf.exp(old_neglog_pac_ph - neglogpac)
        pg_losses = -advs_ph * ratio
        pg_losses2 = -advs_ph * tf.clip_by_value(ratio, 1.0 - clip_range_ph, 1.0 + clip_range_ph)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - old_neglog_pac_ph))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), clip_range_ph)))
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        with tf.variable_scope('model'):
            params = tf.trainable_variables()
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=learning_rate_ph, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)

        def train(learning_rate, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
            """
            Training of PPO2 Algorithm

            :param learning_rate: (float) learning rate
            :param cliprange: (float) Clipping factor
            :param obs: (numpy array) The current observation of the environment
            :param returns: (numpy array) the rewards
            :param masks: (numpy array) The last masks for done episodes (used in recurent policies)
            :param actions: (numpy array) the actions
            :param values: (numpy array) the values
            :param neglogpacs: (numpy array) Negative Log-likelihood probability of Actions
            :param states: (numpy array) For recurrent policies, the internal state of the recurrent model
            :return: policy gradient loss, value function loss, policy entropy,
                    approximation of kl divergence, updated clipping range, training update operation
            """
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {train_model.obs_ph: obs, action_ph: actions, advs_ph: advs, rewards_ph: returns,
                      learning_rate_ph: learning_rate, clip_range_ph: cliprange, old_neglog_pac_ph: neglogpacs,
                      old_vpred_ph: values}
            if states is not None:
                td_map[train_model.states_ph] = states
                td_map[train_model.masks_ph] = masks
            return sess.run([pg_loss, vf_loss, entropy, approxkl, clipfrac, _train], td_map)[:-1]

        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        def save(save_path):
            """
            Save the policy to a file

            :param save_path: (str) the location to save the policy
            """
            saved_params = sess.run(params)
            joblib.dump(saved_params, save_path)

        def load(load_path):
            """
            load a policy from the file

            :param load_path: (str) the saved location of the policy
            """
            loaded_params = joblib.load(load_path)
            restores = []
            for param, loaded_p in zip(params, loaded_params):
                restores.append(param.assign(loaded_p))
            sess.run(restores)
            # If you want to load weights, also save/load observation scaling inside VecNormalize

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)  # pylint: disable=E1101


class Runner(AbstractEnvRunner):
    def __init__(self, *, env, model, n_steps, gamma, lam):
        """
        A runner to learn the policy of an environment for a model

        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        :param gamma: (float) Discount factor
        :param lam: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        """
        super().__init__(env=env, model=model, n_steps=n_steps)
        self.lam = lam
        self.gamma = gamma

    def run(self):
        """
        Run a learning step of the model

        :return:
            - observations: (numpy Number) the observations
            - rewards: (numpy Number) the rewards
            - masks: (numpy bool) whether an episode is over or not
            - actions: (numpy Number) the actions
            - values: (numpy Number) the value function output
            - negative log probabilities: (numpy Number)
            - states: (numpy Number) the internal states of the recurrent policies
            - infos: (dict) the extra information of the model
        """
        # mb stands for minibatch
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], []
        mb_states = self.states
        ep_infos = []
        for _ in range(self.n_steps):
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeep_info = info.get('episode')
                if maybeep_info:
                    ep_infos.append(maybeep_info)
            mb_rewards.append(rewards)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)
        # discount/bootstrap off value fn
        mb_advs = np.zeros_like(mb_rewards)
        last_gae_lam = 0
        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[step + 1]
                nextvalues = mb_values[step + 1]
            delta = mb_rewards[step] + self.gamma * nextvalues * nextnonterminal - mb_values[step]
            mb_advs[step] = last_gae_lam = delta + self.gamma * self.lam * nextnonterminal * last_gae_lam
        mb_returns = mb_advs + mb_values
        return (*map(swap_and_flatten, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)), mb_states,
                ep_infos)


# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def swap_and_flatten(arr):
    """
    swap and then flatten axes 0 and 1

    :param arr: (numpy array)
    :return: (numpy array)
    """
    shape = arr.shape
    return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])


def constfn(val):
    """
    Create a function that returns a constant
    It is useful for learning rate schedule (to avoid code duplication)

    :param val: (float)
    :return: (function)
    """

    def func(_):
        return val

    return func


def learn(*, policy, env, n_steps, total_timesteps, ent_coef, learning_rate,
          vf_coef=0.5, max_grad_norm=0.5, gamma=0.99, lam=0.95,
          log_interval=10, nminibatches=4, noptepochs=4,
          cliprange=0.2, save_interval=0, load_path=None):
    """
    Return a trained PPO2 model.

    :param policy: (A2CPolicy) The policy model to use (MLP, CNN, LSTM, ...)
    :param env: (Gym environment) The environment to learn from
    :param n_steps: (int) The number of steps to run for each environment
    :param total_timesteps: (int) The total number of samples
    :param ent_coef: (float) Entropy coefficient for the loss caculation
    :param learning_rate: (float or callable) The learning rate, it can be a function
    :param vf_coef: (float) Value function coefficient for the loss calculation
    :param max_grad_norm: (float) The maximum value for the gradient clipping
    :param gamma: (float) Discount factor
    :param lam: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param nminibatches: (int) Number of minibatches for the policies
    :param noptepochs: (int) Number of epoch when optimizing the surrogate
    :param cliprange: (float or callable) Clipping parameter, it can be a function
    :param log_interval: (int) The number of timesteps before logging.
    :param save_interval: (int) The number of timesteps before saving.
    :param load_path: (str) Path to a trained ppo2 model, set to None, it will learn from scratch
    :return: (Model) PPO2 model
    """
    if isinstance(learning_rate, float):
        learning_rate = constfn(learning_rate)
    else:
        assert callable(learning_rate)
    if isinstance(cliprange, float):
        cliprange = constfn(cliprange)
    else:
        assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    n_envs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    n_batch = n_envs * n_steps
    n_batch_train = n_batch // nminibatches

    make_model = lambda: Model(policy=policy, ob_space=ob_space, ac_space=ac_space, n_batch_act=n_envs,
                               n_batch_train=n_batch_train, n_steps=n_steps, ent_coef=ent_coef, vf_coef=vf_coef,
                               max_grad_norm=max_grad_norm)
    if save_interval and logger.get_dir():
        import cloudpickle
        with open(os.path.join(logger.get_dir(), 'make_model.pkl'), 'wb') as file_handler:
            file_handler.write(cloudpickle.dumps(make_model))
    model = make_model()
    if load_path is not None:
        model.load(load_path)
    runner = Runner(env=env, model=model, n_steps=n_steps, gamma=gamma, lam=lam)

    ep_info_buf = deque(maxlen=100)
    t_first_start = time.time()

    nupdates = total_timesteps // n_batch
    for update in range(1, nupdates + 1):
        assert n_batch % nminibatches == 0
        n_batch_train = n_batch // nminibatches
        t_start = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lr_now = learning_rate(frac)
        cliprangenow = cliprange(frac)
        obs, returns, masks, actions, values, neglogpacs, states, ep_infos = runner.run()  # pylint: disable=E0632
        ep_info_buf.extend(ep_infos)
        mb_loss_vals = []
        if states is None:  # nonrecurrent version
            inds = np.arange(n_batch)
            for _ in range(noptepochs):
                np.random.shuffle(inds)
                for start in range(0, n_batch, n_batch_train):
                    end = start + n_batch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mb_loss_vals.append(model.train(lr_now, cliprangenow, *slices))
        else:  # recurrent version
            assert n_envs % nminibatches == 0
            envinds = np.arange(n_envs)
            flatinds = np.arange(n_envs * n_steps).reshape(n_envs, n_steps)
            envsperbatch = n_batch_train // n_steps
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, n_envs, envsperbatch):
                    end = start + envsperbatch
                    mb_env_inds = envinds[start:end]
                    mb_flat_inds = flatinds[mb_env_inds].ravel()
                    slices = (arr[mb_flat_inds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mb_states = states[mb_env_inds]
                    mb_loss_vals.append(model.train(lr_now, cliprangenow, *slices, mb_states))

        loss_vals = np.mean(mb_loss_vals, axis=0)
        t_now = time.time()
        fps = int(n_batch / (t_now - t_start))
        if update % log_interval == 0 or update == 1:
            explained_var = explained_variance(values, returns)
            logger.logkv("serial_timesteps", update * n_steps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update * n_batch)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(explained_var))
            logger.logkv('ep_rewmean', safe_mean([ep_info['r'] for ep_info in ep_info_buf]))
            logger.logkv('eplenmean', safe_mean([ep_info['l'] for ep_info in ep_info_buf]))
            logger.logkv('time_elapsed', t_start - t_first_start)
            for (loss_val, loss_name) in zip(loss_vals, model.loss_names):
                logger.logkv(loss_name, loss_val)
            logger.dumpkvs()
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            checkdir = os.path.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            save_path = os.path.join(checkdir, '%.5i' % update)
            print('Saving to', save_path)
            model.save(save_path)
    env.close()
    return model


def safe_mean(arr):
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return zero. It is used for logging only.

    :param arr: (numpy array)
    :return: (float)
    """
    return np.nan if len(arr) == 0 else np.mean(arr)
