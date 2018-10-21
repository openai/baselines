import gym
import baselines.contract
import numpy as np


class ContractEnv(gym.Wrapper):
    def __init__(self, env, contracts):
        gym.Wrapper.__init__(self, env)
        self.contracts = contracts

    def reset(self, **kwargs):
        [c.reset() for c in self.contracts]
        ob = np.array([self.env.reset(**kwargs), [c.state_id() for c in self.contracts]])
        return ob

    def step(self, action):
        ob, rew, done, info = self.env.step(action)
        for c in self.contracts:
            is_vio = c.step(action)
            if is_vio: rew += c.violation_reward
            info[c.name + '-epviols'] = c.epviols
            info[c.name + '-eprmod'] = c.eprmod
        ob = np.array([ob, [c.state_id() for c in self.contracts]])
        return ob, rew, done, info
