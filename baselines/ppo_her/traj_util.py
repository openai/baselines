import random


class Trajectory(object):

    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.done = False

    def __len__(self):
        return len(self.actions)

    def append(self, step):
        """ append step """
        obs, action, reward = step
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)

    def pop(self):
        """ pop step """
        return self.obs.pop(), self.actions.pop(), self.rewards.pop()

    def update_obs(self, key, value):
        for obs in self.obs:
            obs[key] = value

    def update_rewards(self, compute_reward):
        assert callable(compute_reward)
        for t in range(len(self)):
            action, obs = self.actions[t], self.obs[t+1]
            self.rewards[t] = compute_reward(obs['achieved_goal'], obs['desired_goal'], {'action': action})[0]


class TrajectoriesBuffer(object):

    def __init__(self, size):
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
