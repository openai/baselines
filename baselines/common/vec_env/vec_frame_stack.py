from baselines.common.vec_env import VecEnv
import numpy as np
from gym import spaces

class VecFrameStack(VecEnv):
    """
    Vectorized environment base class
    """
    def __init__(self, venv, nstack):
        self.venv = venv
        self.nstack = nstack
        wos = venv.observation_space # wrapped ob space
        low = np.repeat(wos.low, self.nstack, axis=-1)
        high = np.repeat(wos.high, self.nstack, axis=-1)
        self.stackedobs = np.zeros((venv.num_envs,)+low.shape, low.dtype)
        self._observation_space = spaces.Box(low=low, high=high)
        self._action_space = venv.action_space
    def step(self, vac):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)

        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs, rews, news, infos = self.venv.step(vac)
        self.stackedobs = np.roll(self.stackedobs, shift=-1, axis=-1)
        for (i, new) in enumerate(news):
            if new:
                self.stackedobs[i] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs, rews, news, infos
    def reset(self):
        """
        Reset all environments
        """
        obs = self.venv.reset()
        self.stackedobs[...] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs
    @property
    def action_space(self):
        return self._action_space
    @property
    def observation_space(self):
        return self._observation_space
    def close(self):
        self.venv.close()
    @property
    def num_envs(self):
        return self.venv.num_envs