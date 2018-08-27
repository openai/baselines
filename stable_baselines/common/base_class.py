from abc import ABC, abstractmethod
import os

import cloudpickle
import numpy as np
import gym

from stable_baselines.common import set_global_seeds
from stable_baselines.common.policies import LstmPolicy
from stable_baselines.common.vec_env import VecEnvWrapper, VecEnv, DummyVecEnv
from stable_baselines import logger


class BaseRLModel(ABC):
    """
    The base RL model

    :param policy: (Object) Policy object
    :param env: (Gym environment) The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param requires_vec_env: (bool)
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    """

    def __init__(self, policy, env, requires_vec_env, verbose=0):
        super(BaseRLModel, self).__init__()

        self.policy = policy
        self.env = env
        self.verbose = verbose
        self._requires_vec_env = requires_vec_env
        self.observation_space = None
        self.action_space = None
        self.n_envs = None
        self._vectorize_action = False

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
                        self.n_envs = 1
                        self._vectorize_action = True
                    else:
                        raise ValueError("Error: the model requires a non vectorized environment or a single vectorized"
                                         " environment.")

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

        # for models that dont want vectorized environment, check if they make sense and adapt them.
        # Otherwise tell the user about this issue-
        if not self._requires_vec_env and isinstance(env, VecEnv):
            if env.num_envs == 1:
                env = _UnvecWrapper(env)
                self.n_envs = 1
                self._vectorize_action = True
            else:
                raise ValueError("Error: the model requires a non vectorized environment or a single vectorized "
                                 "environment.")
        else:
            self._vectorize_action = False

        self.env = env

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
    def learn(self, total_timesteps, callback=None, seed=None, log_interval=100):
        """
        Return a trained model.

        :param total_timesteps: (int) The total number of samples to train on
        :param seed: (int) The initial seed for training, if None: keep current seed
        :param callback: (function (dict, dict)) function called at every steps with state of the algorithm.
            It takes the local and global variables.
        :param log_interval: (int) The number of timesteps before logging.
        :return: (BaseRLModel) the trained model
        """
        pass

    @abstractmethod
    def predict(self, observation, state=None, mask=None):
        """
        Get the model's action from an observation

        :param observation: (numpy Number) the input observation
        :param state: (numpy Number) The last states (can be None, used in reccurent policies)
        :param mask: (numpy Number) The last masks (can be None, used in reccurent policies)
        :return: (numpy Number, numpy Number) the model's action and the next state (used in reccurent policies)
        """
        pass

    @abstractmethod
    def action_probability(self, observation, state=None, mask=None):
        """
        Get the model's action probability distribution from an observation

        :param observation: (numpy Number) the input observation
        :param state: (numpy Number) The last states (can be None, used in reccurent policies)
        :param mask: (numpy Number) The last masks (can be None, used in reccurent policies)
        :return: (numpy Number) the model's action probability distribution
        """
        pass

    @abstractmethod
    def save(self, save_path):
        """
        Save the current parameters to file

        :param save_path: (str) the save location
        """
        # self._save_to_file(save_path, data={}, params=None)
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def load(cls, load_path, env=None, **kwargs):
        """
        Load the model from file

        :param load_path: (str) the saved parameter location
        :param env: (Gym Envrionment) the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model)
        :param kwargs: extra arguments to change the model when loading
        """
        # data, param = cls._load_from_file(load_path)
        raise NotImplementedError()

    @staticmethod
    def _save_to_file(save_path, data=None, params=None):
        _, ext = os.path.splitext(save_path)
        if ext == "":
            save_path += ".pkl"

        with open(save_path, "wb") as file:
            cloudpickle.dump((data, params), file)

    @staticmethod
    def _load_from_file(load_path):
        if not os.path.exists(load_path):
            if os.path.exists(load_path + ".pkl"):
                load_path += ".pkl"
            else:
                raise ValueError("Error: the file {} could not be found".format(load_path))

        with open(load_path, "rb") as file:
            data, params = cloudpickle.load(file)

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
        return actions[0], values[0], states[0], information[0]

    def render(self, mode='human'):
        return self.venv.render(mode)[0]


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
