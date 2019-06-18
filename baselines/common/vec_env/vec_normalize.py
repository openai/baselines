from . import VecEnvWrapper
from baselines.common.running_mean_std import RunningMeanStd
import numpy as np
from gym.spaces import Dict


class VecNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        VecEnvWrapper.__init__(self, venv)
        if isinstance(self.observation_space, Dict):
            self.ob_rms = {}
            for key in self.observation_space.spaces.keys():
                self.ob_rms[key] = RunningMeanStd(shape=self.observation_space.spaces[key].shape) if ob else None
        else:
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
        self.ret[news] = 0.
        return obs, rews, news, infos

    def _obfilt(self, obs):
        def _obfilt(obs, ob_rms):
            if ob_rms:
                ob_rms.update(obs)
                obs = np.clip((obs - ob_rms.mean) / np.sqrt(ob_rms.var + self.epsilon), -self.clipob, self.clipob)
                return obs
            else:
                return obs

        if isinstance(self.ob_rms, dict):
            for key in self.ob_rms:
                obs[key] = _obfilt(obs[key], self.ob_rms[key])
        else:
            obs = _obfilt(obs, self.ob_rms)

        return obs

    def reset(self):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset()
        return self._obfilt(obs)
