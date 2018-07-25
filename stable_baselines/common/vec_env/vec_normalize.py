import numpy as np

from stable_baselines.common.vec_env import VecEnvWrapper
from stable_baselines.common.running_mean_std import RunningMeanStd


class VecNormalize(VecEnvWrapper):
    def __init__(self, venv, norm_obs=True, norm_reward=True,
                 clip_obs=10., clip_reward=10., gamma=0.99, epsilon=1e-8):
        """
        A rolling average, normalizing, vectorized wrapepr for environment base class
        
        :param venv: ([Gym Environment]) the list of environments to vectorize and normalize
        :param norm_obs: (bool) normalize observation
        :param norm_reward: (bool) normalize reward with discounting (r = sum(r_old) * gamma + r_new)
        :param clip_obs: (float) clipping value for nomalizing observation
        :param clip_reward: (float) clipping value for nomalizing reward
        :param gamma: (float) discount factor
        :param epsilon: (float) epsilon value to avoid arithmetic issues
        """
        VecEnvWrapper.__init__(self, venv)
        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if norm_obs else None
        self.ret_rms = RunningMeanStd(shape=()) if norm_reward else None
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rewards
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rewards = np.clip(rewards / np.sqrt(self.ret_rms.var + self.epsilon), -self.clip_reward, self.clip_reward)
        return obs, rewards, dones, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon),
                          -self.clip_obs, self.clip_obs)
            return obs
        else:
            return obs

    def reset(self):
        obs = self.venv.reset()
        return self._obfilt(obs)
