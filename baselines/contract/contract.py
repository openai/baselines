from baselines.contract.dfa import DFA

class Contract(DFA):
    def __init__(self, reg_ex, violation_reward):
        super(Contract, self).__init__(reg_ex)
        self.violation_reward = violation_reward

class ContractSet(object):
    def __init__(self, names, regexes, rewards):
        self.contracts = {n: Contract(reg, rew) for (n, reg, rew) in zip(names, regexes, rewards)}

    def step(self, action):
        return [c.step(action) for c in self.contracts.values()]

    def reset(self):
        [c.reset() for c in self.contracts.values()]

    def get_violation_rewards(self):
        return [c.violation_reward for c in self.contracts.values()]

    def get_contract_names(self):
        return self.contracts.keys()