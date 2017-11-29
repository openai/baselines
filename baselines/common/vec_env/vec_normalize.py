from baselines.common.vec_env import VecEnv
from baselines.common.running_mean_std import RunningMeanStd
import numpy as np

class VecNormalize(VecEnv):
    """
    Vectorized environment base class
    """
    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        self.venv = venv
        self._observation_space = self.venv.observation_space
        self._action_space = venv.action_space
        self.ob_rms = RunningMeanStd(shape=self._observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
    def step(self, vac):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)

        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs, rews, news, infos = self.venv.step(vac)
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
        """
        Reset all environments
        """
        obs = self.venv.reset()
        return self._obfilt(obs)
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



class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.zeros(shape, 'float64')
        self.count = epsilon


    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count        
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count        

def test_runningmeanstd():
    for (x1, x2, x3) in [
        (np.random.randn(3), np.random.randn(4), np.random.randn(5)),
        (np.random.randn(3,2), np.random.randn(4,2), np.random.randn(5,2)),
        ]:

        rms = RunningMeanStd(epsilon=0.0, shape=x1.shape[1:])

        x = np.concatenate([x1, x2, x3], axis=0)
        ms1 = [x.mean(axis=0), x.var(axis=0)]
        rms.update(x1)
        rms.update(x2)
        rms.update(x3)
        ms2 = [rms.mean, rms.var]

        assert np.allclose(ms1, ms2)
