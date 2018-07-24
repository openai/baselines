from abc import ABC, abstractmethod
import os

import cloudpickle
import numpy as np

from baselines.common import set_global_seeds


class BaseRLModel(ABC):
    def __init__(self, env, requires_vec_env, verbose=0):
        """
        The base RL model

        :param env: (Gym environment) The environment to learn from (can be None for loading trained models)
        :param requires_vec_env: (bool)
        :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
        """
        super(BaseRLModel, self).__init__()

        self.env = env
        self.verbose = verbose
        self._requires_vec_env = requires_vec_env
        self.observation_space = None
        self.action_space = None
        self.n_envs = None

        if env is not None:
            self.observation_space = env.observation_space
            self.action_space = env.action_space
            if requires_vec_env:
                if hasattr(env, "num_envs"):
                    self.n_envs = env.num_envs
                else:
                    raise ValueError("Error: the model requires a vectorized environment, please use a VecEnv wrapper.")

    def set_env(self, env):
        """
        Checks the validity of the environment, and if it is coherent, set it as the current environment.

        :param env: (Gym Environment) The environment for learning a policy
        """
        if env is None and self.env is None and self.verbose == 1:
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
            assert hasattr(env, "num_envs"), \
                "Error: the environment passed is not a vectorized environment, however {} requires it".format(
                    self.__class__.__name__)
            assert self.n_envs == env.num_envs, \
                "Error: the environment passed must have the same number of environments as the model was trained on."

        self.env = env

    def setup_model(self):
        """
        Create all the functions and tensorflow graphs necessary to train the model
        """
        if self.verbose <= 1:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    def _setup_learn(self, seed):
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
    def _softmax(x):
        """
        An implementation of softmax.

        :param x: (numpy float) input vector
        :return: (numpy float) output vector
        """
        e_x = np.exp(x.T - np.max(x.T, axis=0))
        return (e_x / e_x.sum(axis=0)).T

