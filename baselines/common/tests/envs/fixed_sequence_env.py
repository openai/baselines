import numpy as np
from gym import Env
from gym.spaces import Discrete


class FixedSequenceEnv(Env):
    def __init__(
            self,
            n_actions=10,
            seed=0,
            episode_len=100
    ):
        self.np_random = np.random.RandomState()
        self.np_random.seed(seed)
        self.sequence = [self.np_random.randint(0, n_actions-1) for _ in range(episode_len)]

        self.action_space = Discrete(n_actions)
        self.observation_space = Discrete(1)

        self.episode_len = episode_len
        self.time = 0
        self.reset()

    def reset(self):
        self.time = 0
        return 0

    def step(self, actions):
        rew = self._get_reward(actions)
        self._choose_next_state()
        done = False
        if self.episode_len and self.time >= self.episode_len:
            rew = 0
            done = True

        return 0, rew, done, {}

    def _choose_next_state(self):
        self.time += 1

    def _get_reward(self, actions):
        return 1 if actions == self.sequence[self.time] else 0


