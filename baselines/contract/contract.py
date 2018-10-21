from baselines.contract.dfa import DFA

class Contract(DFA):
    def __init__(self, name, reg_ex, violation_reward):
        super(Contract, self).__init__(reg_ex)
        self.name = name
        self.violation_reward = violation_reward
        self.epviols = 0
        self.eprmod = 0.

    def step(self, action):
        is_v = super().step(action)
        if is_v:
            self.epviols += 1
            self.eprmod += self.violation_reward
        return is_v

    def reset(self):
        self.epviols = 0
        self.eprmod = 0.
        return super().reset()
