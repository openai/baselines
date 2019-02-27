import numpy as np
from abc import abstractmethod
from gym import Env
from gym.spaces import MultiDiscrete, Discrete, Box


class IdentityEnv(Env):
    def __init__(
            self,
            episode_len=None
    ):

        self.observation_space = self.action_space
        self.episode_len = episode_len
        self.time = 0
        self.reset()

    def reset(self):
        self._choose_next_state()
        self.time = 0

        return self.state

    def step(self, actions):
        rew = self._get_reward(actions)
        self._choose_next_state()
        done = False
        if self.episode_len and self.time >= self.episode_len:
            done = True

        return self.state, rew, done, {}

    def seed(self, seed=None):
        self.action_space.seed(seed)

    def _choose_next_state(self):
        self.state = self.action_space.sample()
        self.time += 1

    @abstractmethod
    def _get_reward(self, actions):
        raise NotImplementedError


class DiscreteIdentityEnv(IdentityEnv):
    def __init__(
            self,
            dim,
            episode_len=None,
    ):

        self.action_space = Discrete(dim)
        super().__init__(episode_len=episode_len)

    def _get_reward(self, actions):
        return 1 if self.state == actions else 0

class MultiDiscreteIdentityEnv(IdentityEnv):
    def __init__(
            self,
            dims,
            episode_len=None,
    ):

        self.action_space = MultiDiscrete(dims)
        super().__init__(episode_len=episode_len)

    def _get_reward(self, actions):
        return 1 if all(self.state == actions) else 0


class BoxIdentityEnv(IdentityEnv):
    def __init__(
            self,
            shape,
            episode_len=None,
    ):

        self.action_space = Box(low=-1.0, high=1.0, shape=shape, dtype=np.float32)
        super().__init__(episode_len=episode_len)

    def _get_reward(self, actions):
        diff = actions - self.state
        diff = diff[:]
        return -0.5 * np.dot(diff, diff)
