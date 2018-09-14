from abc import ABC, abstractmethod
import os
import glob

import cloudpickle
import numpy as np
import gym
import tensorflow as tf

from stable_baselines.common import set_global_seeds
from stable_baselines.common.policies import LstmPolicy, get_policy_from_name
from stable_baselines.common.vec_env import VecEnvWrapper, VecEnv, DummyVecEnv
from stable_baselines import logger


class BaseRLModel(ABC):
    """
    The base RL model

    :param policy: (Object) Policy object
    :param env: (Gym environment) The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param requires_vec_env: (bool)
    :param policy_base: (BasePolicy) the base policy used by this method
    """

    def __init__(self, policy, env, verbose=0, *, requires_vec_env, policy_base):
        super(BaseRLModel, self).__init__()

        if isinstance(policy, str):
            self.policy = get_policy_from_name(policy_base, policy)
        else:
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
    def learn(self, total_timesteps, callback=None, seed=None, log_interval=100, tb_log_name="run"):
        """
        Return a trained model.

        :param total_timesteps: (int) The total number of samples to train on
        :param seed: (int) The initial seed for training, if None: keep current seed
        :param callback: (function (dict, dict)) function called at every steps with state of the algorithm.
            It takes the local and global variables.
        :param log_interval: (int) The number of timesteps before logging.
        :param tb_log_name: (str) the name of the run for tensorboard log
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
    def action_probability(self, observation, state=None, mask=None):
        """
        Get the model's action probability distribution from an observation

        :param observation: (np.ndarray) the input observation
        :param state: (np.ndarray) The last states (can be None, used in recurrent policies)
        :param mask: (np.ndarray) The last masks (can be None, used in recurrent policies)
        :return: (np.ndarray) the model's action probability distribution
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
        return actions[0], float(values[0]), states[0], information[0]

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


class TensorboardWriter:
    def __init__(self, graph, tensorboard_log_path, tb_log_name):
        """
        Create a Tensorboard writer for a code segment, and saves it to the log directory as its own run

        :param graph: (Tensorflow Graph) the model graph
        :param tensorboard_log_path: (str) the save path for the log (can be None for no logging)
        :param tb_log_name: (str) the name of the run for tensorboard log
        """
        self.graph = graph
        self.tensorboard_log_path = tensorboard_log_path
        self.tb_log_name = tb_log_name
        self.writer = None

    def __enter__(self):
        if self.tensorboard_log_path is not None:
            save_path = os.path.join(self.tensorboard_log_path,
                                     "{}_{}".format(self.tb_log_name, self._get_latest_run_id() + 1))
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
