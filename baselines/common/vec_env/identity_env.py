from gym import Env
from gym.spaces import Discrete


class IdentityEnv(Env):
    def __init__(
            self,
            dim,
            ep_length=100,
    ):

        self.action_space = Discrete(dim)
        self.reset()

    def reset(self):
        self._choose_next_state()
        self.observation_space = self.action_space

        return self.state

    def step(self, actions):
        rew = self._get_reward(actions)
        self._choose_next_state()
        return self.state, rew, False, {}

    def _choose_next_state(self):
        self.state = self.action_space.sample()

    def _get_reward(self, actions):
        return 1 if self.state == actions else 0
