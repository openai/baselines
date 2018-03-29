import numpy as np


class RingBuffer(object):
    def __init__(self, maxlen, shape, dtype='float32'):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = np.zeros((maxlen,) + shape).astype(dtype)
        self.shape = shape

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        return self.data[(self.start + idxs) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()

        v = np.array(v,dtype='float32').reshape(self.shape)
        # assert v.dtype == 'float32'
        self.data[(self.start + self.length - 1) % self.maxlen] = v


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class Memory(object):
    def __init__(self, limit, action_shape, observation_shape, state_shape):
        self.limit = limit

        self.observations0 = RingBuffer(limit, shape=observation_shape)
        self.actions = RingBuffer(limit, shape=action_shape)
        self.rewards = RingBuffer(limit, shape=(1,))
        self.terminals1 = RingBuffer(limit, shape=(1,))
        self.observations1 = RingBuffer(limit, shape=observation_shape)


        self.states = RingBuffer(limit, shape=state_shape) 
        self.states1 = RingBuffer(limit, shape=state_shape)

        self.goals = RingBuffer(limit, shape=state_shape) # Goal state the system wants to achieve
        self.goal_observations = RingBuffer(limit, shape=observation_shape)  # Render of frame at goal position for second CNN

    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.random_integers(self.nb_entries - 2, size=batch_size)

        obs0_batch = self.observations0.get_batch(batch_idxs)
        obs1_batch = self.observations1.get_batch(batch_idxs)
        action_batch = self.actions.get_batch(batch_idxs)
        reward_batch = self.rewards.get_batch(batch_idxs)
        terminal1_batch = self.terminals1.get_batch(batch_idxs)

        states_batch = self.states.get_batch(batch_idxs)
        states1_batch = self.states1.get_batch(batch_idxs)

        goals_batch = self.goals.get_batch(batch_idxs)
        goal_observations_batch = self.goal_observations.get_batch(batch_idxs)

        result = {
            'obs0': array_min2d(obs0_batch),
            'obs1': array_min2d(obs1_batch),
            'rewards': array_min2d(reward_batch),
            'actions': array_min2d(action_batch),
            'terminals1': array_min2d(terminal1_batch),
            'states': array_min2d(states_batch),
            'states1': array_min2d(states1_batch),
            'goals_batch': array_min2d(goals_batch),
            'goal_observations': array_min2d(goal_observations_batch)
        }
        return result

    def append(self, state, obs0, action, reward, state1, obs1, terminal1, goal, goal_obs,  training=True):
        if not training:
            return
        self.observations0.append(obs0)
        self.actions.append(action)
        self.rewards.append(reward)
        self.observations1.append(obs1)
        self.terminals1.append(terminal1)

        self.states.append(state)
        self.states1.append(state1)

        self.goals.append(goal)
        self.goal_observations.append(goal_obs)


    @property
    def nb_entries(self):
        return len(self.observations0)
