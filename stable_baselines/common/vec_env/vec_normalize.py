import pickle
import warnings

import numpy as np

from stable_baselines.common.vec_env import VecEnvWrapper
from stable_baselines.common.running_mean_std import RunningMeanStd


class VecNormalize(VecEnvWrapper):
    """
    A moving average, normalizing wrapper for vectorized environment.

    It is pickleable which will save moving averages and configuration parameters.
    The wrapped environment `venv` is not saved, and must be restored manually with
    `set_venv` after being unpickled.

    :param venv: (VecEnv) the vectorized environment to wrap
    :param training: (bool) Whether to update or not the moving average
    :param norm_obs: (bool) Whether to normalize observation or not (default: True)
    :param norm_reward: (bool) Whether to normalize rewards or not (default: True)
    :param clip_obs: (float) Max absolute value for observation
    :param clip_reward: (float) Max value absolute for discounted reward
    :param gamma: (float) discount factor
    :param epsilon: (float) To avoid division by zero
    """

    def __init__(self, venv, training=True, norm_obs=True, norm_reward=True,
                 clip_obs=10., clip_reward=10., gamma=0.99, epsilon=1e-8):
        VecEnvWrapper.__init__(self, venv)
        self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.ret_rms = RunningMeanStd(shape=())
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        # Returns: discounted rewards
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.training = training
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.old_obs = np.array([])

    def __getstate__(self):
        """
        Gets state for pickling.

        Excludes self.venv, as in general VecEnv's may not be pickleable."""
        state = self.__dict__.copy()
        # these attributes are not pickleable
        del state['venv']
        del state['class_attributes']
        # these attributes depend on the above and so we would prefer not to pickle
        del state['ret']
        return state

    def __setstate__(self, state):
        """
        Restores pickled state.

        User must call set_venv() after unpickling before using.

        :param state: (dict)"""
        self.__dict__.update(state)
        assert 'venv' not in state
        self.venv = None

    def set_venv(self, venv):
        """
        Sets the vector environment to wrap to venv.

        Also sets attributes derived from this such as `num_env`.

        :param venv: (VecEnv)
        """
        if self.venv is not None:
            raise ValueError("Trying to set venv of already initialized VecNormalize wrapper.")
        VecEnvWrapper.__init__(self, venv)
        if self.obs_rms.mean.shape != self.observation_space.shape:
            raise ValueError("venv is incompatible with current statistics.")
        self.ret = np.zeros(self.num_envs)

    def step_wait(self):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)

        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        self.old_obs = obs
        obs = self._normalize_observation(obs)
        if self.norm_reward:
            if self.training:
                self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.clip_reward, self.clip_reward)
        self.ret[news] = 0
        return obs, rews, news, infos

    def _normalize_observation(self, obs):
        """
        :param obs: (numpy tensor)
        """
        if self.norm_obs:
            if self.training:
                self.obs_rms.update(obs)
            obs = np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon), -self.clip_obs,
                          self.clip_obs)
            return obs
        else:
            return obs

    def get_original_obs(self):
        """
        returns the unnormalized observation

        :return: (numpy float)
        """
        return self.old_obs

    def reset(self):
        """
        Reset all environments
        """
        obs = self.venv.reset()
        if len(np.array(obs).shape) == 1:  # for when num_cpu is 1
            self.old_obs = [obs]
        else:
            self.old_obs = obs
        self.ret = np.zeros(self.num_envs)
        return self._normalize_observation(obs)

    @staticmethod
    def load(load_path, venv):
        """
        Loads a saved VecNormalize object.

        :param load_path: the path to load from.
        :param venv: the VecEnv to wrap.
        :return: (VecNormalize)
        """
        with open(load_path, "rb") as file_handler:
            vec_normalize = pickle.load(file_handler)
        vec_normalize.set_venv(venv)
        return vec_normalize

    def save(self, save_path):
        with open(save_path, "wb") as file_handler:
            pickle.dump(self, file_handler)

    def save_running_average(self, path):
        """
        :param path: (str) path to log dir

        .. deprecated:: 2.9.0
            This function will be removed in a future version
        """
        warnings.warn("Usage of `save_running_average` is deprecated. Please "
                      "use `save` or pickle instead.", DeprecationWarning)
        for rms, name in zip([self.obs_rms, self.ret_rms], ['obs_rms', 'ret_rms']):
            with open("{}/{}.pkl".format(path, name), 'wb') as file_handler:
                pickle.dump(rms, file_handler)

    def load_running_average(self, path):
        """
        :param path: (str) path to log dir

        .. deprecated:: 2.9.0
            This function will be removed in a future version
        """
        warnings.warn("Usage of `load_running_average` is deprecated. Please "
                      "use `load` or pickle instead.", DeprecationWarning)
        for name in ['obs_rms', 'ret_rms']:
            with open("{}/{}.pkl".format(path, name), 'rb') as file_handler:
                setattr(self, name, pickle.load(file_handler))
