import numpy as np

from stable_baselines.common.vec_env import VecEnvWrapper
from stable_baselines.common.running_mean_std import RunningMeanStd


class VecNormalize(VecEnvWrapper):
    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        """
        A rolling average, normalizing, vectorized wrapepr for environment base class
        
        :param venv: ([Gym Environment]) the list of environments to vectorize and normalize
        :param ob: (bool) normalize observation
        :param ret: (bool) normalize reward with discounting (r = sum(r_old) * gamma + r_new)
        :param clipob: (float) clipping value for nomalizing observation
        :param cliprew: (float) clipping value for nomalizing reward
        :param gamma: (float) discount factor
        :param epsilon: (float) epsilon value to avoid arithmetic issues
        """
        VecEnvWrapper.__init__(self, venv)
        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        obs = self.venv.reset()
        return self._obfilt(obs)
