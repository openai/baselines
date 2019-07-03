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
        for t in range(len(self.obs)):
            self.obs[t][key] = value

    def update_rewards(self, compute_reward):
        assert callable(compute_reward)
        for t in range(len(self)):
            action, obs = self.actions[t], self.obs[t+1]
            self.rewards[t] = compute_reward(obs['achieved_goal'], obs['desired_goal'], {'action': action})[0]

    def split(self, k):
        T = len(self)//k
        trajs = []
        i = 0
        traj = Trajectory()
        while i < len(self):
            traj.append((self.obs[i], self.actions[i], self.rewards[i]))
            if (i+1) % T == 0 or i == (len(self)-1):
                traj.obs.append(self.obs[i+1])
                traj.done = True
                trajs.append(traj)
                traj = Trajectory()
            i += 1
        return trajs


