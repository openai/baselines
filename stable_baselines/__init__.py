import gym

from stable_baselines.a2c import A2C
from stable_baselines.acer import ACER
from stable_baselines.acktr import ACKTR
from stable_baselines.ddpg import DDPG
from stable_baselines.deepq import DQN
from stable_baselines.gail import GAIL
from stable_baselines.ppo1 import PPO1
from stable_baselines.ppo2 import PPO2
from stable_baselines.trpo_mpi import TRPO

__version__ = "2.0.0"

if not hasattr(gym.spaces.MultiBinary, '__eq__'):
    def _eq(self, other):
        return self.n == other.n

    gym.spaces.MultiBinary.__eq__ = _eq

if not hasattr(gym.spaces.MultiDiscrete, '__eq__'):
    def _eq(self, other):
        return self.nvec == other.nvec

    gym.spaces.MultiDiscrete.__eq__ = _eq
