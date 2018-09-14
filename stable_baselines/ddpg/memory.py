import numpy as np


class RingBuffer(object):
    def __init__(self, maxlen, shape, dtype='float32'):
        """
        A buffer object, when full restarts at the initial position

        :param maxlen: (int) the max number of numpy objects to store
        :param shape: (tuple) the shape of the numpy objects you want to store
        :param dtype: (str) the name of the type of the numpy object you want to store
        """
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = np.zeros((maxlen,) + shape).astype(dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        """
        get the value at the indexes

        :param idxs: (int or numpy int) the indexes
        :return: (np.ndarray) the stored information in the buffer at the asked positions
        """
        return self.data[(self.start + idxs) % self.maxlen]

    def append(self, var):
        """
        Append an object to the buffer

        :param var: (np.ndarray) the object you wish to add
        """
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = var


def array_min2d(arr):
    """
    cast to np.ndarray, and make sure it is of 2 dim

    :param arr: ([Any]) the array to clean
    :return: (np.ndarray) the cleaned array
    """
    arr = np.array(arr)
    if arr.ndim >= 2:
        return arr
    return arr.reshape(-1, 1)


class Memory(object):
    def __init__(self, limit, action_shape, observation_shape):
        """
        The replay buffer object

        :param limit: (int) the max number of transitions to store
        :param action_shape: (tuple) the action shape
        :param observation_shape: (tuple) the observation shape
        """
        self.limit = limit

        self.observations0 = RingBuffer(limit, shape=observation_shape)
        self.actions = RingBuffer(limit, shape=action_shape)
        self.rewards = RingBuffer(limit, shape=(1,))
        self.terminals1 = RingBuffer(limit, shape=(1,))
        self.observations1 = RingBuffer(limit, shape=observation_shape)

    def sample(self, batch_size):
        """
        sample a random batch from the buffer

        :param batch_size: (int) the number of element to sample for the batch
        :return: (dict) the sampled batch
        """
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.randint(low=1, high=self.nb_entries - 1, size=batch_size)

        obs0_batch = self.observations0.get_batch(batch_idxs)
        obs1_batch = self.observations1.get_batch(batch_idxs)
        action_batch = self.actions.get_batch(batch_idxs)
        reward_batch = self.rewards.get_batch(batch_idxs)
        terminal1_batch = self.terminals1.get_batch(batch_idxs)

        result = {
            'obs0': array_min2d(obs0_batch),
            'obs1': array_min2d(obs1_batch),
            'rewards': array_min2d(reward_batch),
            'actions': array_min2d(action_batch),
            'terminals1': array_min2d(terminal1_batch),
        }
        return result

    def append(self, obs0, action, reward, obs1, terminal1, training=True):
        """
        Append a transition to the buffer

        :param obs0: ([float] or [int]) the last observation
        :param action: ([float]) the action
        :param reward: (float] the reward
        :param obs1: ([float] or [int]) the current observation
        :param terminal1: (bool) is the episode done
        :param training: (bool) is the RL model training or not
        """
        if not training:
            return

        self.observations0.append(obs0)
        self.actions.append(action)
        self.rewards.append(reward)
        self.observations1.append(obs1)
        self.terminals1.append(terminal1)

    @property
    def nb_entries(self):
        return len(self.observations0)
