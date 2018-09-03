import numpy as np
from gym import spaces

from stable_baselines.common.vec_env import VecEnvWrapper


class VecFrameStack(VecEnvWrapper):
    """
    Frame stacking wrapper for vectorized environment

    :param venv: (VecEnv) the vectorized environment to wrap
    :param n_stack: (int) Number of frames to stack
    """
    
    def __init__(self, venv, n_stack):
        self.venv = venv
        self.n_stack = n_stack
        wrapped_obs_space = venv.observation_space
        low = np.repeat(wrapped_obs_space.low, self.n_stack, axis=-1)
        high = np.repeat(wrapped_obs_space.high, self.n_stack, axis=-1)
        self.stackedobs = np.zeros((venv.num_envs,) + low.shape, low.dtype)
        observation_space = spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()
        self.stackedobs = np.roll(self.stackedobs, shift=-observations.shape[-1], axis=-1)
        for i, done in enumerate(dones):
            if done:
                self.stackedobs[i] = 0
        self.stackedobs[..., -observations.shape[-1]:] = observations
        return self.stackedobs, rewards, dones, infos

    def reset(self):
        """
        Reset all environments
        """
        obs = self.venv.reset()
        self.stackedobs[...] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs

    def close(self):
        self.venv.close()
