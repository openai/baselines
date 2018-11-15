from . import VecEnvWrapper
import numpy as np


class VecActionClip(VecEnvWrapper):
    def step_async(self, actions):
        low = self.action_space.low
        high = self.action_space.high
        actions = [np.clip(action, low, high) for action in actions]

        self.venv.step_async(actions)

    def step_wait(self):
        return self.venv.step_wait()

    def reset(self):
        return self.venv.reset()
