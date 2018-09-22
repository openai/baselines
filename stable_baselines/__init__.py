import gym
import numpy as np

from stable_baselines.a2c import A2C
from stable_baselines.acer import ACER
from stable_baselines.acktr import ACKTR
from stable_baselines.ddpg import DDPG
from stable_baselines.deepq import DQN
from stable_baselines.gail import GAIL
from stable_baselines.ppo1 import PPO1
from stable_baselines.ppo2 import PPO2
from stable_baselines.trpo_mpi import TRPO

__version__ = "2.0.1.a0"


# patch Gym spaces to add equality functions, if not implemented
# See https://github.com/openai/gym/issues/1171
if gym.spaces.MultiBinary.__eq__ == object.__eq__:  # by default, all classes have the __eq__ function from object.
    def _eq(self, other):
        return self.n == other.n

    gym.spaces.MultiBinary.__eq__ = _eq

if gym.spaces.MultiDiscrete.__eq__ == object.__eq__:
    def _eq(self, other):
        return np.all(self.nvec == other.nvec)

    gym.spaces.MultiDiscrete.__eq__ = _eq
