import time
from collections import deque
import sys
import multiprocessing

import numpy as np
import tensorflow as tf

from stable_baselines import logger
from stable_baselines.common import explained_variance, BaseRLModel, tf_util, SetVerbosity
from stable_baselines.common.runners import AbstractEnvRunner
from stable_baselines.common.policies import LstmPolicy


class PPO2(BaseRLModel):
    """
    Proximal Policy Optimization algorithm (GPU version).
    Paper: https://arxiv.org/abs/1707.06347

    :param policy: (ActorCriticPolicy) The policy model to use (MLP, CNN, LSTM, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) Discount factor
    :param n_steps: (int) The number of steps to run for each environment
    :param ent_coef: (float) Entropy coefficient for the loss caculation
    :param learning_rate: (float or callable) The learning rate, it can be a function
    :param vf_coef: (float) Value function coefficient for the loss calculation
    :param max_grad_norm: (float) The maximum value for the gradient clipping
    :param lam: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param nminibatches: (int) Number of minibatches for the policies
    :param noptepochs: (int) Number of epoch when optimizing the surrogate
    :param cliprange: (float or callable) Clipping parameter, it can be a function
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    """

    def __init__(self, policy, env, gamma=0.99, n_steps=128, ent_coef=0.01, learning_rate=2.5e-4, vf_coef=0.5,
                 max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2, verbose=0,
                 _init_setup_model=True):
        super(PPO2, self).__init__(policy=policy, env=env, requires_vec_env=True, verbose=verbose)

        if isinstance(learning_rate, float):
            learning_rate = constfn(learning_rate)
        else:
            assert callable(learning_rate)
        if isinstance(cliprange, float):
            cliprange = constfn(cliprange)
        else:
            assert callable(cliprange)

        self.learning_rate = learning_rate
        self.cliprange = cliprange
        self.n_steps = n_steps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.lam = lam
        self.nminibatches = nminibatches
        self.noptepochs = noptepochs

        self.graph = None
        self.sess = None
        self.action_ph = None
        self.advs_ph = None
        self.rewards_ph = None
        self.old_neglog_pac_ph = None
        self.old_vpred_ph = None
        self.learning_rate_ph = None
        self.clip_range_ph = None
        self.entropy = None
        self.vf_loss = None
        self.pg_loss = None
        self.approxkl = None
        self.clipfrac = None
        self.params = None
        self._train = None
        self.loss_names = None
        self.train_model = None
        self.act_model = None
        self.step = None
        self.proba_step = None
        self.value = None
        self.initial_state = None
        self.n_batch = None

        if _init_setup_model:
            self.setup_model()

    def setup_model(self):
        with SetVerbosity(self.verbose):

            self.n_batch = self.n_envs * self.n_steps

            n_cpu = multiprocessing.cpu_count()
            if sys.platform == 'darwin':
                n_cpu //= 2

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.sess = tf_util.make_session(num_cpu=n_cpu, graph=self.graph)

                n_batch_step = None
                n_batch_train = None
                if issubclass(self.policy, LstmPolicy):
                    n_batch_step = self.n_envs
                    n_batch_train = self.n_batch // self.nminibatches

                act_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                        n_batch_step, reuse=False)
                train_model = self.policy(self.sess, self.observation_space, self.action_space,
                                          self.n_envs // self.nminibatches, self.n_steps, n_batch_train,
                                          reuse=True)

                self.action_ph = train_model.pdtype.sample_placeholder([None])
                self.advs_ph = tf.placeholder(tf.float32, [None])
                self.rewards_ph = tf.placeholder(tf.float32, [None])
                self.old_neglog_pac_ph = tf.placeholder(tf.float32, [None])
                self.old_vpred_ph = tf.placeholder(tf.float32, [None])
                self.learning_rate_ph = tf.placeholder(tf.float32, [])
                self.clip_range_ph = tf.placeholder(tf.float32, [])

                neglogpac = train_model.proba_distribution.neglogp(self.action_ph)
                self.entropy = tf.reduce_mean(train_model.proba_distribution.entropy())

                vpred = train_model.value_fn
                vpredclipped = self.old_vpred_ph + tf.clip_by_value(
                    train_model.value_fn - self.old_vpred_ph, - self.clip_range_ph, self.clip_range_ph)
                vf_losses1 = tf.square(vpred - self.rewards_ph)
                vf_losses2 = tf.square(vpredclipped - self.rewards_ph)
                self.vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
                ratio = tf.exp(self.old_neglog_pac_ph - neglogpac)
                pg_losses = -self.advs_ph * ratio
                pg_losses2 = -self.advs_ph * tf.clip_by_value(ratio, 1.0 - self.clip_range_ph, 1.0 + self.clip_range_ph)
                self.pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
                self.approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - self.old_neglog_pac_ph))
                self.clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), self.clip_range_ph)))
                loss = self.pg_loss - self.entropy * self.ent_coef + self.vf_loss * self.vf_coef
                with tf.variable_scope('model'):
                    self.params = tf.trainable_variables()
                grads = tf.gradients(loss, self.params)
                if self.max_grad_norm is not None:
                    grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
                grads = list(zip(grads, self.params))
                trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph, epsilon=1e-5)
                self._train = trainer.apply_gradients(grads)

                self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

                self.train_model = train_model
                self.act_model = act_model
                self.step = act_model.step
                self.proba_step = act_model.proba_step
                self.value = act_model.value
                self.initial_state = act_model.initial_state
                tf.global_variables_initializer().run(session=self.sess)  # pylint: disable=E1101

    def _train_step(self, learning_rate, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
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
        td_map = {self.train_model.obs_ph: obs, self.action_ph: actions, self.advs_ph: advs, self.rewards_ph: returns,
                  self.learning_rate_ph: learning_rate, self.clip_range_ph: cliprange,
                  self.old_neglog_pac_ph: neglogpacs, self.old_vpred_ph: values}
        if states is not None:
            td_map[self.train_model.states_ph] = states
            td_map[self.train_model.masks_ph] = masks
        return self.sess.run([self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train],
                             td_map)[:-1]

    def learn(self, total_timesteps, callback=None, seed=None, log_interval=100):
        with SetVerbosity(self.verbose):
            self._setup_learn(seed)

            runner = Runner(env=self.env, model=self, n_steps=self.n_steps, gamma=self.gamma, lam=self.lam)

            ep_info_buf = deque(maxlen=100)
            t_first_start = time.time()

            nupdates = total_timesteps // self.n_batch
            for update in range(1, nupdates + 1):
                assert self.n_batch % self.nminibatches == 0
                n_batch_train = self.n_batch // self.nminibatches
                t_start = time.time()
                frac = 1.0 - (update - 1.0) / nupdates
                lr_now = self.learning_rate(frac)
                cliprangenow = self.cliprange(frac)
                obs, returns, masks, actions, values, neglogpacs, states, ep_infos = runner.run()  # pylint: disable=E0632
                ep_info_buf.extend(ep_infos)
                mb_loss_vals = []
                if states is None:  # nonrecurrent version
                    inds = np.arange(self.n_batch)
                    for _ in range(self.noptepochs):
                        np.random.shuffle(inds)
                        for start in range(0, self.n_batch, n_batch_train):
                            end = start + n_batch_train
                            mbinds = inds[start:end]
                            slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                            mb_loss_vals.append(self._train_step(lr_now, cliprangenow, *slices))
                else:  # recurrent version
                    assert self.n_envs % self.nminibatches == 0
                    envinds = np.arange(self.n_envs)
                    flatinds = np.arange(self.n_envs * self.n_steps).reshape(self.n_envs, self.n_steps)
                    envsperbatch = n_batch_train // self.n_steps
                    for _ in range(self.noptepochs):
                        np.random.shuffle(envinds)
                        for start in range(0, self.n_envs, envsperbatch):
                            end = start + envsperbatch
                            mb_env_inds = envinds[start:end]
                            mb_flat_inds = flatinds[mb_env_inds].ravel()
                            slices = (arr[mb_flat_inds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                            mb_states = states[mb_env_inds]
                            mb_loss_vals.append(self._train_step(lr_now, cliprangenow, *slices, mb_states))

                loss_vals = np.mean(mb_loss_vals, axis=0)
                t_now = time.time()
                fps = int(self.n_batch / (t_now - t_start))

                if callback is not None:
                    callback(locals(), globals())

                if self.verbose >= 1 and (update % log_interval//100 == 0 or update == 1):
                    explained_var = explained_variance(values, returns)
                    logger.logkv("serial_timesteps", update * self.n_steps)
                    logger.logkv("nupdates", update)
                    logger.logkv("total_timesteps", update * self.n_batch)
                    logger.logkv("fps", fps)
                    logger.logkv("explained_variance", float(explained_var))
                    logger.logkv('ep_rewmean', safe_mean([ep_info['r'] for ep_info in ep_info_buf]))
                    logger.logkv('eplenmean', safe_mean([ep_info['l'] for ep_info in ep_info_buf]))
                    logger.logkv('time_elapsed', t_start - t_first_start)
                    for (loss_val, loss_name) in zip(loss_vals, self.loss_names):
                        logger.logkv(loss_name, loss_val)
                    logger.dumpkvs()

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
            "lam": self.lam,
            "nminibatches": self.nminibatches,
            "noptepochs": self.noptepochs,
            "cliprange": self.cliprange,
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


def safe_mean(arr):
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return zero. It is used for logging only.

    :param arr: (numpy array)
    :return: (float)
    """
    return np.nan if len(arr) == 0 else np.mean(arr)
