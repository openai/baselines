import numpy as np

from gym import Env
from gym.spaces import Discrete, MultiDiscrete, MultiBinary, Box


class IdentityEnv(Env):
    def __init__(self, dim, ep_length=100):
        """
        Identity environment for testing purposes

        :param dim: (int) the size of the dimentions you want to learn
        :param ep_length: (int) the length of each episodes in timesteps
        """
        self.action_space = Discrete(dim)
        self.ep_length = ep_length
        self.reset()

    def reset(self):
        self._choose_next_state()
        self.observation_space = self.action_space
        return self.state

    def step(self, action):
        reward = self._get_reward(action)
        self._choose_next_state()
        return self.state, reward, False, {}

    def _choose_next_state(self):
        self.state = self.action_space.sample()

    def _get_reward(self, action):
        return 1 if self.state == action else 0

    def render(self, mode='human'):
        pass


class IdentityEnvMultiDiscrete(Env):
    def __init__(self, dim, ep_length=100):
        """
        Identity environment for testing purposes

        :param dim: (int) the size of the dimentions you want to learn
        :param ep_length: (int) the length of each episodes in timesteps
        """
        self.action_space = MultiDiscrete([dim, dim])
        self.dim = dim
        self.observation_space = Box(low=0, high=1, shape=(dim * 2,), dtype=int)
        self.ep_length = ep_length
        self.reset()

    def reset(self):
        self._choose_next_state()
        return self.state

    def step(self, action):
        reward = self._get_reward(action)
        self._choose_next_state()
        return self.state, reward, False, {}

    def _choose_next_state(self):
        state = np.zeros(self.dim*2, dtype=int)
        mask = self.action_space.sample()
        state[mask[0]] = 1
        state[mask[1] + self.dim] = 1
        self.state = state

    def _get_reward(self, action):
        return 1 if np.all(self.state == action) else 0

    def render(self, mode='human'):
        pass


class IdentityEnvMultiBinary(Env):
    def __init__(self, dim, ep_length=100):
        """
        Identity environment for testing purposes

        :param dim: (int) the size of the dimentions you want to learn
        :param ep_length: (int) the length of each episodes in timesteps
        """
        self.action_space = MultiBinary(dim)
        self.observation_space = Box(low=0, high=1, shape=(dim,), dtype=int)
        self.ep_length = ep_length
        self.reset()

    def reset(self):
        self._choose_next_state()
        return self.state

    def step(self, action):
        reward = self._get_reward(action)
        self._choose_next_state()
        return self.state, reward, False, {}

    def _choose_next_state(self):
        self.state = self.action_space.sample()

    def _get_reward(self, action):
        return 1 if np.all(self.state == action) else 0

    def render(self, mode='human'):
        pass
