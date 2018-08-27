import tensorflow as tf
import numpy as np
import gym

from stable_baselines import logger, deepq
from stable_baselines.common import tf_util, BaseRLModel, SetVerbosity
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from stable_baselines.deepq.utils import ObservationInput
from stable_baselines.a2c.utils import find_trainable_variables


class DeepQ(BaseRLModel):
    """
    The DQN model class. DQN paper: https://arxiv.org/pdf/1312.5602.pdf

    :param policy: (function (TensorFlow Tensor, int, str, bool): TensorFlow Tensor)
                    the policy that takes the following inputs:
                    - observation_in: (object) the output of observation placeholder
                    - num_actions: (int) number of actions
                    - scope: (str)
                    - reuse: (bool) should be passed to outer variable scope
                    and returns a tensor of shape (batch_size, num_actions) with values of every action.
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) discount factor
    :param learning_rate: (float) learning rate for adam optimizer
    :param buffer_size: (int) size of the replay buffer
    :param exploration_fraction: (float) fraction of entire training period over which the exploration rate is
            annealed
    :param exploration_final_eps: (float) final value of random action probability
    :param train_freq: (int) update the model every `train_freq` steps. set to None to disable printing
    :param batch_size: (int) size of a batched sampled from replay buffer for training
    :param checkpoint_freq: (int) how often to save the model. This is so that the best version is restored at the
            end of the training. If you do not wish to restore the best version
            at the end of the training set this variable to None.
    :param checkpoint_path: (str) replacement path used if you need to log to somewhere else than a temporary
            directory.
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param target_network_update_freq: (int) update the target network every `target_network_update_freq` steps.
    :param prioritized_replay: (bool) if True prioritized replay buffer will be used.
    :param prioritized_replay_alpha: (float) alpha parameter for prioritized replay buffer
    :param prioritized_replay_beta0: (float) initial value of beta for prioritized replay buffer
    :param prioritized_replay_beta_iters: (int) number of iterations over which beta will be annealed from initial
            value to 1.0. If set to None equals to max_timesteps.
    :param prioritized_replay_eps: (float) epsilon to add to the TD errors when updating priorities.
    :param param_noise: (bool) Whether or not to apply noise to the parameters of the policy.
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    """

    def __init__(self, policy, env, gamma=0.99, learning_rate=5e-4, buffer_size=50000, exploration_fraction=0.1,
                 exploration_final_eps=0.02, train_freq=1, batch_size=32, checkpoint_freq=10000, checkpoint_path=None,
                 learning_starts=1000, target_network_update_freq=500, prioritized_replay=False,
                 prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=None,
                 prioritized_replay_eps=1e-6, param_noise=False, verbose=0, _init_setup_model=True):
        super(DeepQ, self).__init__(policy=policy, env=env, requires_vec_env=False, verbose=verbose)

        assert not isinstance(policy, ActorCriticPolicy), \
            "Error: DeepQ does not support the actor critic policies, please use the " \
            "'stable_baselines.deepq.models.mlp' and 'stable_baselines.deepq.models.cnn_to_mlp' " \
            "functions to create your policies."

        self.checkpoint_path = checkpoint_path
        self.param_noise = param_noise
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.prioritized_replay = prioritized_replay
        self.prioritized_replay_eps = prioritized_replay_eps
        self.batch_size = batch_size
        self.target_network_update_freq = target_network_update_freq
        self.checkpoint_freq = checkpoint_freq
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.prioritized_replay_beta0 = prioritized_replay_beta0
        self.prioritized_replay_beta_iters = prioritized_replay_beta_iters
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.graph = None
        self.sess = None
        self._train_step = None
        self.update_target = None
        self.act = None
        self.replay_buffer = None
        self.beta_schedule = None
        self.exploration = None
        self.params = None

        if _init_setup_model:
            self.setup_model()

    def setup_model(self):
        with SetVerbosity(self.verbose):

            assert isinstance(self.action_space, gym.spaces.Discrete), \
                "Error: DeepQ cannot output a {} action space, only spaces.Discrete is supported."\
                .format(self.action_space)

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.sess = tf_util.make_session(graph=self.graph)

                # capture the shape outside the closure so that the env object is not serialized
                # by cloudpickle when serializing make_obs_ph
                observation_space = self.observation_space

                def make_obs_ph(name):
                    """
                    makes the observation placeholder

                    :param name: (str) the placeholder name
                    :return: (TensorFlow Tensor) the placeholder
                    """
                    return ObservationInput(observation_space, name=name)

                self.act, self._train_step, self.update_target, _ = deepq.build_train(
                    make_obs_ph=make_obs_ph,
                    q_func=self.policy,
                    num_actions=self.action_space.n,
                    optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate),
                    gamma=self.gamma,
                    grad_norm_clipping=10,
                    param_noise=self.param_noise
                )

                self.params = find_trainable_variables("deepq")

                # Initialize the parameters and copy them to the target network.
                tf_util.initialize(self.sess)
                self.update_target(sess=self.sess)

    def learn(self, total_timesteps, callback=None, seed=None, log_interval=100):
        with SetVerbosity(self.verbose):
            self._setup_learn(seed)

            # Create the replay buffer
            if self.prioritized_replay:
                self.replay_buffer = PrioritizedReplayBuffer(self.buffer_size, alpha=self.prioritized_replay_alpha)
                if self.prioritized_replay_beta_iters is None:
                    prioritized_replay_beta_iters = total_timesteps
                    self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                                        initial_p=self.prioritized_replay_beta0,
                                                        final_p=1.0)
            else:
                self.replay_buffer = ReplayBuffer(self.buffer_size)
                self.beta_schedule = None
            # Create the schedule for exploration starting from 1.
            self.exploration = LinearSchedule(schedule_timesteps=int(self.exploration_fraction * total_timesteps),
                                              initial_p=1.0,
                                              final_p=self.exploration_final_eps)

            episode_rewards = [0.0]
            obs = self.env.reset()
            reset = True

            for step in range(total_timesteps):
                if callback is not None:
                    callback(locals(), globals())
                # Take action and update exploration to the newest value
                kwargs = {}
                if not self.param_noise:
                    update_eps = self.exploration.value(step)
                    update_param_noise_threshold = 0.
                else:
                    update_eps = 0.
                    # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                    # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                    # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                    # for detailed explanation.
                    update_param_noise_threshold = \
                        -np.log(1. - self.exploration.value(step) +
                                self.exploration.value(step) / float(self.env.action_space.n))
                    kwargs['reset'] = reset
                    kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                    kwargs['update_param_noise_scale'] = True
                with self.sess.as_default():
                    action = self.act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
                env_action = action
                reset = False
                new_obs, rew, done, _ = self.env.step(env_action)
                # Store transition in the replay buffer.
                self.replay_buffer.add(obs, action, rew, new_obs, float(done))
                obs = new_obs

                episode_rewards[-1] += rew
                if done:
                    if not isinstance(self.env, VecEnv):
                        obs = self.env.reset()
                    episode_rewards.append(0.0)
                    reset = True

                if step > self.learning_starts and step % self.train_freq == 0:
                    # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                    if self.prioritized_replay:
                        experience = self.replay_buffer.sample(self.batch_size, beta=self.beta_schedule.value(step))
                        (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                    else:
                        obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(self.batch_size)
                        weights, batch_idxes = np.ones_like(rewards), None
                    td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, dones, weights,
                                                 sess=self.sess)
                    if self.prioritized_replay:
                        new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
                        self.replay_buffer.update_priorities(batch_idxes, new_priorities)

                if step > self.learning_starts and step % self.target_network_update_freq == 0:
                    # Update target network periodically.
                    self.update_target(sess=self.sess)

                if len(episode_rewards[-101:-1]) == 0:
                    mean_100ep_reward = -np.inf
                else:
                    mean_100ep_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

                num_episodes = len(episode_rewards)
                if self.verbose >= 1 and done and log_interval is not None and len(episode_rewards) % log_interval == 0:
                    logger.record_tabular("steps", step)
                    logger.record_tabular("episodes", num_episodes)
                    logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                    logger.record_tabular("% time spent exploring", int(100 * self.exploration.value(step)))
                    logger.dump_tabular()

        return self

    def predict(self, observation, state=None, mask=None):
        observation = np.array(observation).reshape(self.observation_space.shape)

        with self.sess.as_default():
            action = self.act(observation[None])[0]

        if self._vectorize_action:
            return [action], [None]
        else:
            return action, None

    def action_probability(self, observation, state=None, mask=None):
        observation = np.array(observation).reshape((-1,) + self.observation_space.shape)

        # Get the tensor just before the softmax function in the TensorFlow graph,
        # then execute the graph from the input observation to this tensor.
        tensor = self.graph.get_tensor_by_name('deepq/q_func/fully_connected_2/BiasAdd:0')
        if self._vectorize_action:
            return self._softmax(self.sess.run(tensor, feed_dict={'deepq/observation:0': observation}))
        else:
            return self._softmax(self.sess.run(tensor, feed_dict={'deepq/observation:0': observation}))[0]

    def save(self, save_path):
        # params
        data = {
            "checkpoint_path": self.checkpoint_path,
            "param_noise": self.param_noise,
            "learning_starts": self.learning_starts,
            "train_freq": self.train_freq,
            "prioritized_replay": self.prioritized_replay,
            "prioritized_replay_eps": self.prioritized_replay_eps,
            "batch_size": self.batch_size,
            "target_network_update_freq": self.target_network_update_freq,
            "checkpoint_freq": self.checkpoint_freq,
            "prioritized_replay_alpha": self.prioritized_replay_alpha,
            "prioritized_replay_beta0": self.prioritized_replay_beta0,
            "prioritized_replay_beta_iters": self.prioritized_replay_beta_iters,
            "exploration_final_eps": self.exploration_final_eps,
            "exploration_fraction": self.exploration_fraction,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "verbose": self.verbose,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "policy": self.policy,
            "n_envs": self.n_envs,
            "_vectorize_action": self._vectorize_action
        }

        params = self.sess.run(self.params)

        self._save_to_file(save_path, data=data, params=params)

    @classmethod
    def load(cls, load_path, env=None, **kwargs):
        data, params = cls._load_from_file(load_path)

        model = cls(policy=data["policy"], env=env, _init_setup_model=False)
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model.set_env(env)
        model.setup_model()

        restores = []
        for param, loaded_p in zip(model.params, params):
            restores.append(param.assign(loaded_p))
        model.sess.run(restores)

        return model
