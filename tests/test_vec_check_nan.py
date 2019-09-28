import gym
from gym import spaces
import numpy as np

from stable_baselines.common.vec_env import DummyVecEnv, VecCheckNan


class NanAndInfEnv(gym.Env):
    """Custom Environment that raised NaNs and Infs"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(NanAndInfEnv, self).__init__()
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)

    @staticmethod
    def step(action):
        if np.all(np.array(action) > 0):
            obs = float('NaN')
        elif np.all(np.array(action) < 0):
            obs = float('inf')
        else:
            obs = 0
        return [obs], 0.0, False, {}

    @staticmethod
    def reset():
        return [0.0]

    def render(self, mode='human', close=False):
        pass


def test_check_nan():
    """Test VecCheckNan Object"""

    env = DummyVecEnv([NanAndInfEnv])
    env = VecCheckNan(env, raise_exception=True)

    env.step([[0]])

    try:
        env.step([[float('NaN')]])
    except ValueError:
        pass
    else:
        assert False

    try:
        env.step([[float('inf')]])
    except ValueError:
        pass
    else:
        assert False

    try:
        env.step([[-1]])
    except ValueError:
        pass
    else:
        assert False

    try:
        env.step([[1]])
    except ValueError:
        pass
    else:
        assert False


    env.step(np.array([[0, 1], [0, 1]]))
