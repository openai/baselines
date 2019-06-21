import random


class Trajectory(object):

    def __init__(self):
        self.obs = []
        self.actions = []

    def __len__(self):
        return len(self.actions)


class TrajectoriesBuffer(object):

    def __init__(self):
        self.trajectories = []

    def __len__(self):
        len(self.trajectories)

    def add(self, trajectories):
        if isinstance(trajectories, list):
            for trajectory in trajectories:
                assert isinstance(trajectory, Trajectory)
            random.shuffle(trajectories)
            self.trajectories.extend(trajectories)

    def sample(self, k):
        k = k if len(self) > k else len(self)
        if k > 0:
            return random.sample(self.trajectories, k)
        return []
