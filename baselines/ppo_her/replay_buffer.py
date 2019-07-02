from baselines.ppo_her.trajectory import Trajectory
import random


class ReplayBuffer(object):

    def __init__(self, size=0):
        self.size = size
        self.trajectories = []

    def __len__(self):
        return len(self.trajectories)

    def add(self, trajectories):
        if isinstance(trajectories, list):
            for trajectory in trajectories:
                assert isinstance(trajectory, Trajectory)
            random.shuffle(trajectories)
            self.trajectories.extend(trajectories)
        if len(self) > self.size:
            self.trajectories = self.trajectories[-self.size:]

    def sample(self, k):
        k = k if len(self) > k else len(self)
        if k > 0:
            return random.sample(self.trajectories, k)
        return []
