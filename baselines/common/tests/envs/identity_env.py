import numpy as np
from abc import abstractmethod
from gym import Env
from gym.spaces import MultiDiscrete, Discrete, Box
from collections import deque

class IdentityEnv(Env):
    def __init__(
            self,
            episode_len=None,
            delay=0,
            zero_first_rewards=True
    ):

        self.observation_space = self.action_space
        self.episode_len = episode_len
        self.time = 0
        self.delay = delay
        self.zero_first_rewards = zero_first_rewards
        self.q = deque(maxlen=delay+1)

    def reset(self):
        self.q.clear()
        for _ in range(self.delay + 1):
            self.q.append(self.action_space.sample())
        self.time = 0

        return self.q[-1]

    def step(self, actions):
        rew = self._get_reward(self.q.popleft(), actions)
        if self.zero_first_rewards and self.time < self.delay:
            rew = 0
        self.q.append(self.action_space.sample())
        self.time += 1
        done = self.episode_len is not None and self.time >= self.episode_len
        return self.q[-1], rew, done, {}

    def seed(self, seed=None):
        self.action_space.seed(seed)

    @abstractmethod
    def _get_reward(self, state, actions):
        raise NotImplementedError


class DiscreteIdentityEnv(IdentityEnv):
    def __init__(
            self,
            dim,
            episode_len=None,
            delay=0,
            zero_first_rewards=True
    ):

        self.action_space = Discrete(dim)
        super().__init__(episode_len=episode_len, delay=delay, zero_first_rewards=zero_first_rewards)

    def _get_reward(self, state, actions):
        return 1 if state == actions else 0

class MultiDiscreteIdentityEnv(IdentityEnv):
    def __init__(
            self,
            dims,
            episode_len=None,
            delay=0,
    ):

        self.action_space = MultiDiscrete(dims)
        super().__init__(episode_len=episode_len, delay=delay)

    def _get_reward(self, state, actions):
        return 1 if all(state == actions) else 0


class BoxIdentityEnv(IdentityEnv):
    def __init__(
            self,
            shape,
            episode_len=None,
    ):

        self.action_space = Box(low=-1.0, high=1.0, shape=shape, dtype=np.float32)
        super().__init__(episode_len=episode_len)

    def _get_reward(self, state, actions):
        diff = actions - state
        diff = diff[:]
        return -0.5 * np.dot(diff, diff)
