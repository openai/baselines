from abc import ABC, abstractmethod
import os
import glob
import warnings

import cloudpickle
import numpy as np
import gym
import tensorflow as tf

from stable_baselines.common import set_global_seeds
from stable_baselines.common.policies import LstmPolicy, get_policy_from_name, ActorCriticPolicy
from stable_baselines.common.vec_env import VecEnvWrapper, VecEnv, DummyVecEnv
from stable_baselines import logger


class BaseRLModel(ABC):
    """
    The base RL model

    :param policy: (BasePolicy) Policy object
    :param env: (Gym environment) The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param requires_vec_env: (bool) Does this model require a vectorized environment
    :param policy_base: (BasePolicy) the base policy used by this method
    """

    def __init__(self, policy, env, verbose=0, *, requires_vec_env, policy_base, policy_kwargs=None):
        if isinstance(policy, str):
            self.policy = get_policy_from_name(policy_base, policy)
        else:
            self.policy = policy
        self.env = env
        self.verbose = verbose
        self._requires_vec_env = requires_vec_env
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        self.observation_space = None
        self.action_space = None
        self.n_envs = None
        self._vectorize_action = False
        self.num_timesteps = 0
        self.graph = None
        self.sess = None
        self.params = None

        if env is not None:
            if isinstance(env, str):
                if self.verbose >= 1:
                    print("Creating environment from the given name, wrapped in a DummyVecEnv.")
                self.env = env = DummyVecEnv([lambda: gym.make(env)])

            self.observation_space = env.observation_space
            self.action_space = env.action_space
            if requires_vec_env:
                if isinstance(env, VecEnv):
                    self.n_envs = env.num_envs
                else:
                    raise ValueError("Error: the model requires a vectorized environment, please use a VecEnv wrapper.")
            else:
                if isinstance(env, VecEnv):
                    if env.num_envs == 1:
                        self.env = _UnvecWrapper(env)
                        self._vectorize_action = True
                    else:
                        raise ValueError("Error: the model requires a non vectorized environment or a single vectorized"
                                         " environment.")
                self.n_envs = 1

    def get_env(self):
        """
        returns the current environment (can be None if not defined)

        :return: (Gym Environment) The current environment
        """
        return self.env

    def set_env(self, env):
        """
        Checks the validity of the environment, and if it is coherent, set it as the current environment.

        :param env: (Gym Environment) The environment for learning a policy
        """
        if env is None and self.env is None:
            if self.verbose >= 1:
                print("Loading a model without an environment, "
                      "this model cannot be trained until it has a valid environment.")
            return
        elif env is None:
            raise ValueError("Error: trying to replace the current environment with None")

        # sanity checking the environment
        assert self.observation_space == env.observation_space, \
            "Error: the environment passed must have at least the same observation space as the model was trained on."
        assert self.action_space == env.action_space, \
            "Error: the environment passed must have at least the same action space as the model was trained on."
        if self._requires_vec_env:
            assert isinstance(env, VecEnv), \
                "Error: the environment passed is not a vectorized environment, however {} requires it".format(
                    self.__class__.__name__)
            assert not issubclass(self.policy, LstmPolicy) or self.n_envs == env.num_envs, \
                "Error: the environment passed must have the same number of environments as the model was trained on." \
                "This is due to the Lstm policy not being capable of changing the number of environments."
            self.n_envs = env.num_envs
        else:
            # for models that dont want vectorized environment, check if they make sense and adapt them.
            # Otherwise tell the user about this issue
            if isinstance(env, VecEnv):
                if env.num_envs == 1:
                    env = _UnvecWrapper(env)
                    self._vectorize_action = True
                else:
                    raise ValueError("Error: the model requires a non vectorized environment or a single vectorized "
                                     "environment.")
            else:
                self._vectorize_action = False

            self.n_envs = 1

        self.env = env

    def _init_num_timesteps(self, reset_num_timesteps=True):
        """
        Initialize and resets num_timesteps (total timesteps since beginning of training)
        if needed. Mainly used logging and plotting (tensorboard).

        :param reset_num_timesteps: (bool) Set it to false when continuing training
            to not create new plotting curves in tensorboard.
        :return: (bool) Whether a new tensorboard log needs to be created
        """
        if reset_num_timesteps:
            self.num_timesteps = 0

        new_tb_log = self.num_timesteps == 0
        return new_tb_log

    @abstractmethod
    def setup_model(self):
        """
        Create all the functions and tensorflow graphs necessary to train the model
        """
        pass

    def _setup_learn(self, seed):
        """
        check the environment, set the seed, and set the logger

        :param seed: (int) the seed value
        """
        if self.env is None:
            raise ValueError("Error: cannot train the model without a valid environment, please set an environment with"
                             "set_env(self, env) method.")
        if seed is not None:
            set_global_seeds(seed)

    @abstractmethod
    def _get_pretrain_placeholders(self):
        """
        Return the placeholders needed for the pretraining:
        - obs_ph: observation placeholder
        - actions_ph will be population with an action from the environement
            (from the expert dataset)
        - deterministic_actions_ph: e.g., in the case of a gaussian policy,
            the mean.

        :return: ((tf.placeholder)) (obs_ph, actions_ph, deterministic_actions_ph)
        """
        pass

    def pretrain(self, dataset, n_epochs=10, learning_rate=1e-4,
                 adam_epsilon=1e-8, val_interval=None):
        """
        Pretrain a model using behavior cloning:
        supervised learning given an expert dataset.

        NOTE: only Box and Discrete spaces are supported for now.

        :param dataset: (ExpertDataset) Dataset manager
        :param n_epochs: (int) Number of iterations on the training set
        :param learning_rate: (float) Learning rate
        :param adam_epsilon: (float) the epsilon value for the adam optimizer
        :param val_interval: (int) Report training and validation losses every n epochs.
            By default, every 10th of the maximum number of epochs.
        :return: (BaseRLModel) the pretrained model
        """
        continuous_actions = isinstance(self.action_space, gym.spaces.Box)
        discrete_actions = isinstance(self.action_space, gym.spaces.Discrete)

        assert discrete_actions or continuous_actions, 'Only Discrete and Box action spaces are supported'

        # Validate the model every 10% of the total number of iteration
        if val_interval is None:
            # Prevent modulo by zero
            if n_epochs < 10:
                val_interval = 1
            else:
                val_interval = int(n_epochs / 10)

        with self.graph.as_default():
            with tf.variable_scope('pretrain'):
                if continuous_actions:
                    obs_ph, actions_ph, deterministic_actions_ph = self._get_pretrain_placeholders()
                    loss = tf.reduce_mean(tf.square(actions_ph - deterministic_actions_ph))
                else:
                    obs_ph, actions_ph, actions_logits_ph = self._get_pretrain_placeholders()
                    # actions_ph has a shape if (n_batch,), we reshape it to (n_batch, 1)
                    # so no additional changes is needed in the dataloader
                    actions_ph = tf.expand_dims(actions_ph, axis=1)
                    one_hot_actions = tf.one_hot(actions_ph, self.action_space.n)
                    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                        logits=actions_logits_ph,
                        labels=tf.stop_gradient(one_hot_actions)
                    )
                    loss = tf.reduce_mean(loss)
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=adam_epsilon)
                optim_op = optimizer.minimize(loss, var_list=self.params)

            self.sess.run(tf.global_variables_initializer())

        if self.verbose > 0:
            print("Pretraining with Behavior Cloning...")

        for epoch_idx in range(int(n_epochs)):
            train_loss = 0.0
            # Full pass on the training set
            for _ in range(len(dataset.train_loader)):
                expert_obs, expert_actions = dataset.get_next_batch('train')
                feed_dict = {
                    obs_ph: expert_obs,
                    actions_ph: expert_actions,
                }
                train_loss_, _ = self.sess.run([loss, optim_op], feed_dict)
                train_loss += train_loss_

            train_loss /= len(dataset.train_loader)

            if self.verbose > 0 and (epoch_idx + 1) % val_interval == 0:
                val_loss = 0.0
                # Full pass on the validation set
                for _ in range(len(dataset.val_loader)):
                    expert_obs, expert_actions = dataset.get_next_batch('val')
                    val_loss_, = self.sess.run([loss], {obs_ph: expert_obs,
                                                        actions_ph: expert_actions})
                    val_loss += val_loss_

                val_loss /= len(dataset.val_loader)
                if self.verbose > 0:
                    print("==== Training progress {:.2f}% ====".format(100 * (epoch_idx + 1) / n_epochs))
                    print('Epoch {}'.format(epoch_idx + 1))
                    print("Training loss: {:.6f}, Validation loss: {:.6f}".format(train_loss, val_loss))
                    print()
            # Free memory
            del expert_obs, expert_actions
        if self.verbose > 0:
            print("Pretraining done.")
        return self

    @abstractmethod
    def learn(self, total_timesteps, callback=None, seed=None, log_interval=100, tb_log_name="run",
              reset_num_timesteps=True):
        """
        Return a trained model.

        :param total_timesteps: (int) The total number of samples to train on
        :param seed: (int) The initial seed for training, if None: keep current seed
        :param callback: (function (dict, dict)) -> boolean function called at every steps with state of the algorithm.
            It takes the local and global variables. If it returns False, training is aborted.
        :param log_interval: (int) The number of timesteps before logging.
        :param tb_log_name: (str) the name of the run for tensorboard log
        :param reset_num_timesteps: (bool) whether or not to reset the current timestep number (used in logging)
        :return: (BaseRLModel) the trained model
        """
        pass

    @abstractmethod
    def predict(self, observation, state=None, mask=None, deterministic=False):
        """
        Get the model's action from an observation

        :param observation: (np.ndarray) the input observation
        :param state: (np.ndarray) The last states (can be None, used in recurrent policies)
        :param mask: (np.ndarray) The last masks (can be None, used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: (np.ndarray, np.ndarray) the model's action and the next state (used in recurrent policies)
        """
        pass

    @abstractmethod
    def action_probability(self, observation, state=None, mask=None, actions=None):
        """
        If ``actions`` is ``None``, then get the model's action probability distribution from a given observation

        depending on the action space the output is:
            - Discrete: probability for each possible action
            - Box: mean and standard deviation of the action output

        However if ``actions`` is not ``None``, this function will return the probability that the given actions are
        taken with the given parameters (observation, state, ...) on this model.

        .. warning::
            When working with continuous probability distribution (e.g. Gaussian distribution for continuous action)
            the probability of taking a particular action is exactly zero.
            See http://blog.christianperone.com/2019/01/ for a good explanation

        :param observation: (np.ndarray) the input observation
        :param state: (np.ndarray) The last states (can be None, used in recurrent policies)
        :param mask: (np.ndarray) The last masks (can be None, used in recurrent policies)
        :param actions: (np.ndarray) (OPTIONAL) For calculating the likelihood that the given actions are chosen by
            the model for each of the given parameters. Must have the same number of actions and observations.
            (set to None to return the complete action probability distribution)
        :return: (np.ndarray) the model's action probability
        """
        pass

    @abstractmethod
    def save(self, save_path):
        """
        Save the current parameters to file

        :param save_path: (str or file-like object) the save location
        """
        # self._save_to_file(save_path, data={}, params=None)
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def load(cls, load_path, env=None, **kwargs):
        """
        Load the model from file

        :param load_path: (str or file-like) the saved parameter location
        :param env: (Gym Envrionment) the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model)
        :param kwargs: extra arguments to change the model when loading
        """
        # data, param = cls._load_from_file(load_path)
        raise NotImplementedError()

    @staticmethod
    def _save_to_file(save_path, data=None, params=None):
        if isinstance(save_path, str):
            _, ext = os.path.splitext(save_path)
            if ext == "":
                save_path += ".pkl"

            with open(save_path, "wb") as file_:
                cloudpickle.dump((data, params), file_)
        else:
            # Here save_path is a file-like object, not a path
            cloudpickle.dump((data, params), save_path)

    @staticmethod
    def _load_from_file(load_path):
        if isinstance(load_path, str):
            if not os.path.exists(load_path):
                if os.path.exists(load_path + ".pkl"):
                    load_path += ".pkl"
                else:
                    raise ValueError("Error: the file {} could not be found".format(load_path))

            with open(load_path, "rb") as file:
                data, params = cloudpickle.load(file)
        else:
            # Here load_path is a file-like object, not a path
            data, params = cloudpickle.load(load_path)

        return data, params

    @staticmethod
    def _softmax(x_input):
        """
        An implementation of softmax.

        :param x_input: (numpy float) input vector
        :return: (numpy float) output vector
        """
        x_exp = np.exp(x_input.T - np.max(x_input.T, axis=0))
        return (x_exp / x_exp.sum(axis=0)).T

    @staticmethod
    def _is_vectorized_observation(observation, observation_space):
        """
        For every observation type, detects and validates the shape,
        then returns whether or not the observation is vectorized.

        :param observation: (np.ndarray) the input observation to validate
        :param observation_space: (gym.spaces) the observation space
        :return: (bool) whether the given observation is vectorized or not
        """
        if isinstance(observation_space, gym.spaces.Box):
            if observation.shape == observation_space.shape:
                return False
            elif observation.shape[1:] == observation_space.shape:
                return True
            else:
                raise ValueError("Error: Unexpected observation shape {} for ".format(observation.shape) +
                                 "Box environment, please use {} ".format(observation_space.shape) +
                                 "or (n_env, {}) for the observation shape."
                                 .format(", ".join(map(str, observation_space.shape))))
        elif isinstance(observation_space, gym.spaces.Discrete):
            if observation.shape == ():  # A numpy array of a number, has shape empty tuple '()'
                return False
            elif len(observation.shape) == 1:
                return True
            else:
                raise ValueError("Error: Unexpected observation shape {} for ".format(observation.shape) +
                                 "Discrete environment, please use (1,) or (n_env, 1) for the observation shape.")
        elif isinstance(observation_space, gym.spaces.MultiDiscrete):
            if observation.shape == (len(observation_space.nvec),):
                return False
            elif len(observation.shape) == 2 and observation.shape[1] == len(observation_space.nvec):
                return True
            else:
                raise ValueError("Error: Unexpected observation shape {} for MultiDiscrete ".format(observation.shape) +
                                 "environment, please use ({},) or ".format(len(observation_space.nvec)) +
                                 "(n_env, {}) for the observation shape.".format(len(observation_space.nvec)))
        elif isinstance(observation_space, gym.spaces.MultiBinary):
            if observation.shape == (observation_space.n,):
                return False
            elif len(observation.shape) == 2 and observation.shape[1] == observation_space.n:
                return True
            else:
                raise ValueError("Error: Unexpected observation shape {} for MultiBinary ".format(observation.shape) +
                                 "environment, please use ({},) or ".format(observation_space.n) +
                                 "(n_env, {}) for the observation shape.".format(observation_space.n))
        else:
            raise ValueError("Error: Cannot determine if the observation is vectorized with the space type {}."
                             .format(observation_space))


class ActorCriticRLModel(BaseRLModel):
    """
    The base class for Actor critic model

    :param policy: (BasePolicy) Policy object
    :param env: (Gym environment) The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param policy_base: (BasePolicy) the base policy used by this method (default=ActorCriticPolicy)
    :param requires_vec_env: (bool) Does this model require a vectorized environment
    """

    def __init__(self, policy, env, _init_setup_model, verbose=0, policy_base=ActorCriticPolicy,
                 requires_vec_env=False, policy_kwargs=None):
        super(ActorCriticRLModel, self).__init__(policy, env, verbose=verbose, requires_vec_env=requires_vec_env,
                                                 policy_base=policy_base, policy_kwargs=policy_kwargs)

        self.sess = None
        self.initial_state = None
        self.step = None
        self.proba_step = None
        self.params = None

    @abstractmethod
    def setup_model(self):
        pass

    @abstractmethod
    def learn(self, total_timesteps, callback=None, seed=None,
              log_interval=100, tb_log_name="run", reset_num_timesteps=True):
        pass

    def predict(self, observation, state=None, mask=None, deterministic=False):
        if state is None:
            state = self.initial_state
        if mask is None:
            mask = [False for _ in range(self.n_envs)]
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions, _, states, _ = self.step(observation, state, mask, deterministic=deterministic)

        clipped_actions = actions
        # Clip the actions to avoid out of bound error
        if isinstance(self.action_space, gym.spaces.Box):
            clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            clipped_actions = clipped_actions[0]

        return clipped_actions, states

    def action_probability(self, observation, state=None, mask=None, actions=None):
        if state is None:
            state = self.initial_state
        if mask is None:
            mask = [False for _ in range(self.n_envs)]
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions_proba = self.proba_step(observation, state, mask)

        if len(actions_proba) == 0:  # empty list means not implemented
            warnings.warn("Warning: action probability is not implemented for {} action space. Returning None."
                          .format(type(self.action_space).__name__))
            return None

        if actions is not None:  # comparing the action distribution, to given actions
            actions = np.array([actions])
            if isinstance(self.action_space, gym.spaces.Discrete):
                actions = actions.reshape((-1,))
                assert observation.shape[0] == actions.shape[0], \
                    "Error: batch sizes differ for actions and observations."
                actions_proba = actions_proba[np.arange(actions.shape[0]), actions]

            elif isinstance(self.action_space, gym.spaces.MultiDiscrete):
                actions = actions.reshape((-1, len(self.action_space.nvec)))
                assert observation.shape[0] == actions.shape[0], \
                    "Error: batch sizes differ for actions and observations."
                # Discrete action probability, over multiple categories
                actions = np.swapaxes(actions, 0, 1)  # swap axis for easier categorical split
                actions_proba = np.prod([proba[np.arange(act.shape[0]), act]
                                         for proba, act in zip(actions_proba, actions)], axis=0)

            elif isinstance(self.action_space, gym.spaces.MultiBinary):
                actions = actions.reshape((-1, self.action_space.n))
                assert observation.shape[0] == actions.shape[0], \
                    "Error: batch sizes differ for actions and observations."
                # Bernoulli action probability, for every action
                actions_proba = np.prod(actions_proba * actions + (1 - actions_proba) * (1 - actions), axis=1)

            elif isinstance(self.action_space, gym.spaces.Box):
                warnings.warn("The probabilty of taken a given action is exactly zero for a continuous distribution."
                              "See http://blog.christianperone.com/2019/01/ for a good explanation")
                actions_proba = np.zeros((observation.shape[0], 1), dtype=np.float32)
            else:
                warnings.warn("Warning: action_probability not implemented for {} actions space. Returning None."
                              .format(type(self.action_space).__name__))
                return None
            # normalize action proba shape for the different gym spaces
            actions_proba = actions_proba.reshape((-1, 1))

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            actions_proba = actions_proba[0]

        return actions_proba

    @abstractmethod
    def save(self, save_path):
        pass

    @classmethod
    def load(cls, load_path, env=None, **kwargs):
        data, params = cls._load_from_file(load_path)

        if 'policy_kwargs' in kwargs and kwargs['policy_kwargs'] != data['policy_kwargs']:
            raise ValueError("The specified policy kwargs do not equal the stored policy kwargs. "
                             "Stored kwargs: {}, specified kwargs: {}".format(data['policy_kwargs'],
                                                                              kwargs['policy_kwargs']))

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


class OffPolicyRLModel(BaseRLModel):
    """
    The base class for off policy RL model

    :param policy: (BasePolicy) Policy object
    :param env: (Gym environment) The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param replay_buffer: (ReplayBuffer) the type of replay buffer
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param requires_vec_env: (bool) Does this model require a vectorized environment
    :param policy_base: (BasePolicy) the base policy used by this method
    """

    def __init__(self, policy, env, replay_buffer, verbose=0, *, requires_vec_env, policy_base, policy_kwargs=None):
        super(OffPolicyRLModel, self).__init__(policy, env, verbose=verbose, requires_vec_env=requires_vec_env,
                                               policy_base=policy_base, policy_kwargs=policy_kwargs)

        self.replay_buffer = replay_buffer

    @abstractmethod
    def setup_model(self):
        pass

    @abstractmethod
    def learn(self, total_timesteps, callback=None, seed=None,
              log_interval=100, tb_log_name="run", reset_num_timesteps=True):
        pass

    @abstractmethod
    def predict(self, observation, state=None, mask=None, deterministic=False):
        pass

    @abstractmethod
    def action_probability(self, observation, state=None, mask=None, actions=None):
        pass

    @abstractmethod
    def save(self, save_path):
        pass

    @classmethod
    @abstractmethod
    def load(cls, load_path, env=None, **kwargs):
        pass


class _UnvecWrapper(VecEnvWrapper):
    def __init__(self, venv):
        """
        Unvectorize a vectorized environment, for vectorized environment that only have one environment

        :param venv: (VecEnv) the vectorized environment to wrap
        """
        super().__init__(venv)
        assert venv.num_envs == 1, "Error: cannot unwrap a environment wrapper that has more than one environment."

    def reset(self):
        return self.venv.reset()[0]

    def step_async(self, actions):
        self.venv.step_async([actions])

    def step_wait(self):
        actions, values, states, information = self.venv.step_wait()
        return actions[0], float(values[0]), states[0], information[0]

    def render(self, mode='human'):
        return self.venv.render(mode=mode)


class SetVerbosity:
    def __init__(self, verbose=0):
        """
        define a region of code for certain level of verbosity

        :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
        """
        self.verbose = verbose

    def __enter__(self):
        self.tf_level = os.environ.get('TF_CPP_MIN_LOG_LEVEL', '0')
        self.log_level = logger.get_level()
        self.gym_level = gym.logger.MIN_LEVEL

        if self.verbose <= 1:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        if self.verbose <= 0:
            logger.set_level(logger.DISABLED)
            gym.logger.set_level(gym.logger.DISABLED)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verbose <= 1:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = self.tf_level

        if self.verbose <= 0:
            logger.set_level(self.log_level)
            gym.logger.set_level(self.gym_level)


class TensorboardWriter:
    def __init__(self, graph, tensorboard_log_path, tb_log_name, new_tb_log=True):
        """
        Create a Tensorboard writer for a code segment, and saves it to the log directory as its own run

        :param graph: (Tensorflow Graph) the model graph
        :param tensorboard_log_path: (str) the save path for the log (can be None for no logging)
        :param tb_log_name: (str) the name of the run for tensorboard log
        :param new_tb_log: (bool) whether or not to create a new logging folder for tensorbaord
        """
        self.graph = graph
        self.tensorboard_log_path = tensorboard_log_path
        self.tb_log_name = tb_log_name
        self.writer = None
        self.new_tb_log = new_tb_log

    def __enter__(self):
        if self.tensorboard_log_path is not None:
            latest_run_id = self._get_latest_run_id()
            if self.new_tb_log:
                latest_run_id = latest_run_id + 1
            save_path = os.path.join(self.tensorboard_log_path, "{}_{}".format(self.tb_log_name, latest_run_id))
            self.writer = tf.summary.FileWriter(save_path, graph=self.graph)
        return self.writer

    def _get_latest_run_id(self):
        """
        returns the latest run number for the given log name and log path,
        by finding the greatest number in the directories.

        :return: (int) latest run number
        """
        max_run_id = 0
        for path in glob.glob(self.tensorboard_log_path + "/{}_[0-9]*".format(self.tb_log_name)):
            file_name = path.split("/")[-1]
            ext = file_name.split("_")[-1]
            if self.tb_log_name == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
                max_run_id = int(ext)
        return max_run_id

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.writer is not None:
            self.writer.add_graph(self.graph)
            self.writer.flush()
