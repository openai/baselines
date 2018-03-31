import numpy as np
from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree

def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)

class Memory(object):
    def __init__(self, limit):
        self._storage = []
        self._maxsize = limit
        self._next_idx = 0
        self._adding_demonstrations = False
        self._num_demonstrations = 0
        self.storable_elements = ["states0", "obs0", "actions", "rewards", "states1", "obs1", "terminals1", "goals", "goal_observations", "aux0", "aux1"]

    def __len__(self):
        return len(self._storage)

    @property
    def nb_entries(self):
        return len(self._storage)

    def append(self, *args,  training=True):
        assert len(args) == len(self.storable_elements)
        if not training:
            return
        entry = args
        if self._next_idx >= len(self._storage):
            self._storage.append(entry)
        else:
            self._storage[self._next_idx] = entry

        self._next_idx = int(self._next_idx + 1)
        if self._next_idx == self._maxsize:
            self._next_idx = self._num_demonstrations

    def _get_batches_for_idxes(self, idxes):
        batches = {storable_element: [] for storable_element in self.storable_elements}
        for i in idxes:
            entry = self._storage[i]
            assert len(entry) == len(self.storable_elements)
            for j, data in enumerate(entry):
                batches[self.storable_elements[j]].append(data)
        result = {k: array_min2d(v) for k, v in batches.items()}
        return result
    def sample(self, batch_size):
        idxes = np.random.random_integers(low=0, high=self.nb_entries - 1, size=batch_size)

        return self._get_batches_for_idxes(idxes)



class PrioritizedMemory(Memory):
    def __init__(self, limit, alpha, transition_small_epsilon=1e-6, demonstartion_epsilon=0.1):

        super(PrioritizedMemory, self).__init__(limit)
        assert alpha > 0
        self._alpha = alpha
        self._transition_small_epsilon = transition_small_epsilon
        self._demonstartion_epsilon = demonstartion_epsilon
        self._print_counter = 0

        self._first_time_sample = True
        self._indexes = []

        it_capacity = 1
        while it_capacity < self._maxsize:
            it_capacity *= 2 # Size must be power of 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0


    def append(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().append(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def append_demonstartion(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha
        self._num_demonstrations += 1

    def _sample_proportional(self, batch_size, pretrain=False):
        res = []
        for _ in range(batch_size):
            while True:
                mass = np.random.uniform(0,self._it_sum.sum(0, len(self._storage) - 1))
                idx = self._it_sum.find_prefixsum_idx(mass)
                if idx not in res:
                    res.append(idx)
                    break
        return res

    def sample(self, batch_size, beta):
        assert beta > 0
        idxes = self._sample_proportional(batch_size)
        demos = [i < self._num_demonstrations for i in idxes]
        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)
        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._get_batches_for_idxes(idxes)
        encoded_sample['weights'] = array_min2d(weights)
        encoded_sample['idxes'] = idxes
        return encoded_sample

    def update_priorities(self, idxes, td_errors, actor_losses=0.0):
        priorities = (td_errors ** 2) + (actor_losses ** 2) + self._transition_small_epsilon
        for i in range(len(priorities)):
            if idxes[i] < self._num_demonstrations:
                priorities[i] += self._demonstartion_epsilon
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha
        self._max_priority = max(self._max_priority, priority)
