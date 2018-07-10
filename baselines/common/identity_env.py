from gym import Env
from gym.spaces import Discrete


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

    def step(self, actions):
        rew = self._get_reward(actions)
        self._choose_next_state()
        return self.state, rew, False, {}

    def _choose_next_state(self):
        self.state = self.action_space.sample()

    def _get_reward(self, actions):
        return 1 if self.state == actions else 0

    def render(self, mode='human'):
        pass
