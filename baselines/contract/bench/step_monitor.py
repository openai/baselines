import numpy as np
import os

from gym.core import Wrapper


class LogBuffer(object):
    def __init__(self, buffer_size, buffer_shape, dtype=np.uint8):
        self.buffer = np.zeros((buffer_size, ) + buffer_shape, dtype)
        self.next_step = 0

    def log(self, item):
        try:
            self.buffer[self.next_step] = item
        except IndexError:
            self.buffer = np.concatenate(
                [self.buffer, np.zeros_like(self.buffer)])
            self.buffer[self.next_step] = item
        self.next_step += 1
        return self.next_step

    def save(self, name):
        np.save(name, self.buffer[:self.next_step - 1])

class StepMonitor(Wrapper):
    def __init__(self, env, filename, log_size=10000):
        Wrapper.__init__(self, env=env)
        self.filename = filename
        self.action_log = LogBuffer(log_size, (), dtype=np.int32)
        self.reward_log = LogBuffer(log_size, (), dtype=np.float32)
        self.done_log = LogBuffer(log_size, (), dtype=np.int32)

    def step(self, action):
        ob, rew, done, info = self.env.step(action)
        self.update(ob, action, rew, done, info)
        return (ob, rew, done, info)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def update(self, ob, act, rew, done, info):
        act_ns = self.action_log.log(act)
        rew_ns = self.reward_log.log(rew)
        don_ns = self.done_log.log(done)
        # assert that the logs are staying in step
        assert act_ns == rew_ns
        assert act_ns == don_ns

        if done: self.save()

    def save(self):
        self.action_log.save(os.path.join(self.filename, 'action'))
        self.reward_log.save(os.path.join(self.filename, 'reward'))
        self.done_log.save(os.path.join(self.filename, 'done'))
