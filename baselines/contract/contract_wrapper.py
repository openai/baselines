import os

import numpy as np

import baselines.contract
import gym
from baselines.contract.bench.step_monitor import LogBuffer


class ContractEnv(gym.Wrapper):
    def __init__(self,
                 env,
                 contracts,
                 augmentation_type=None,
                 log_dir=None):
        gym.Wrapper.__init__(self, env)
        self.contracts = contracts
        self.augmentation_type = augmentation_type
        if log_dir is not None:
            self.log_dir = log_dir
            self.log_dict = dict([(c, LogBuffer(1000, (), dtype=np.bool))
                             for c in contracts])
        else:
            self.logs = None

    def reset(self, **kwargs):
        [c.reset() for c in self.contracts]
        [
            log.save(os.path.join(self.log_dir, c.name))
            for (c, log) in self.log_dict.items()
        ]

        ob = self.env.reset(**kwargs)
        if self.augmentation_type == 'contract_state':
            ob = np.array([ob, [c.state_id() for c in self.contracts]])
        return ob

    def step(self, action):
        ob, rew, done, info = self.env.step(action)
        for c in self.contracts:
            is_vio = c.step(action)
            if is_vio: rew += c.violation_reward
            if self.log_dict is not None:
                self.log_dict[c].log(is_vio)

        if self.augmentation_type == 'contract_state':
            ob = np.array([ob, [c.state_id() for c in self.contracts]])

        return ob, rew, done, info
