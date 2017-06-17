import heapq
import numpy as np
import random

from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class ProportionalReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        """Create Prioritized Proportional Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)

        See Also
        --------
        ReplayBuffer.__init__
        """
        super(ProportionalReplayBuffer, self).__init__(size)
        assert alpha > 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        """See ReplayBuffer.add"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            # TODO(szymon): should we ensure no repeats?
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.


        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min()

        for idx in idxes:
            p_sample = self._it_sum[idx]
            weight_normalized = (p_sample / p_min) ** (-beta)
            weights.append(weight_normalized)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)



class RankBasedReplayBuffer(ReplayBuffer):
    class HeapNode(object):
        def __init__(self, priority, data):
            """Create a node for efficient heap storage.

            Parameters
            ----------
            priority: float
                The priority of the node. Used for comparisons within the heap.
            data: list
                See ReplayBuffer.add.

            """
            self.priority = priority
            self.data = data

        def __lt__(self, other):
            return self.priority < other.priority

    def __init__(self, size, alpha, n_segs, sort_period=10000, segs_recompute_thresh=0.01):
        """Create Prioritized Rank-based Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the
            buffer overflows the memories with low rank are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        n_segs: int
            Used to precompute segments for better performance.
            See RankBasedReplayBuffer._compute_segment_boundaries.
        sort_period: int
            Sort the heap every `sort_period` steps. Trade-off between
            accuracy for lower values and performance for higher values.
        segs_recompute_thresh: float
            Segments of equal probability density are precomputed for
            more efficient sampling. This parameter specifies the
            percentage threshold of `max_size` increase required for
            recomputing the segments.

        See Also
        --------
        ReplayBuffer.__init__
        """
        super(RankBasedReplayBuffer, self).__init__(size)

        assert alpha > 0
        assert 0 < segs_recompute_thresh < 1
        self._alpha = alpha

        self._n_segs = n_segs
        self._segs_start_idxes = None
        self._segs_recompute_thresh = segs_recompute_thresh
        self._segs_computed_buffer_size = len(self._storage)
        self._sort_period = sort_period
        self._steps_to_sort = sort_period
        self._max_priority = None
        self._sum_priorities = 0

    def add(self, obs_t, action, reward, obs_tp1, done):
        if self._max_priority is not None:
            max_priority = self._max_priority ** self._alpha
        else:
            max_priority = 1.0
        if len(self._storage) < self._maxsize:
            self._sum_priorities += (len(self._storage) + 1) ** (-self._alpha)
        else:
            del self._storage[-1]
        # Priority negated since the heap is implemented as min-heap.
        data = self.HeapNode(-max_priority, [obs_t, action, reward, obs_tp1, done])

        heapq.heappush(self._storage, data)

        # Check if segments need to be recomputed.
        currsize = len(self._storage)
        if currsize - self._segs_computed_buffer_size >= \
                self._segs_recompute_thresh * self._maxsize or \
                currsize == 10 * self._n_segs or \
                currsize == self._maxsize and self._segs_computed_buffer_size < currsize:
            self._compute_segment_boundaries()

        # Check if heap needs to be sorted.
        self._steps_to_sort -= 1
        if self._steps_to_sort <= 0:
            self._steps_to_sort = self._sort_period
            self._storage.sort()

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            # Ignore the weight.
            obs_t, action, reward, obs_tp1, done = data.data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def _compute_segment_boundaries(self):
        """Computes boundaries of segments.

        The segments are computed such that each has
        the same probability density.
        """
        if len(self._storage) == 0:
            # Repetition.
            return [0 for i in range(self._n_segs)]

        cdf_average = 1.0 / self._n_segs
        idx = 0
        self._segs_start_idxes = [0]
        density = 0.0
        for segment in range(1, self._n_segs):
            while density < segment * cdf_average:
                density += (idx + 1) ** (-self._alpha) / self._sum_priorities
                idx += 1
            self._segs_start_idxes.append(idx)
        self._segs_computed_buffer_size = len(self._storage)

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.


        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0
        if self._segs_start_idxes is None:
            # Segments not yet computed. Return uniformly random samples.
            idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        else:
            idxes = []
            for _ in range(batch_size):
                idx = random.randint(0, len(self._segs_start_idxes) - 1)
                low = self._segs_start_idxes[idx]
                if idx < len(self._segs_start_idxes) - 1:
                    high = self._segs_start_idxes[idx + 1]
                else:
                    high = len(self._storage)
                idxes.append(random.randint(low, high - 1))
            idxes = np.array(idxes)

        encoded_sample = self._encode_sample(idxes)

        weights = []
        p_min = 1.0 / len(self._storage)
        for idx in idxes:
            p_sample = 1.0 / (idx + 1)
            weight_normalized = (p_sample / p_min) ** (-beta)
            weights.append(weight_normalized)

        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i]. After the priorities are updated, the
        order of elements in the buffer changes.

        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        # Prepare for update.
        if self._max_priority is None:
            self._max_priority = -np.inf

        new_samples = []

        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            new_samples.append(self.HeapNode(-priority ** self._alpha, self._storage[idx].data))
            self._storage[idx].priority = -np.inf
            self._max_priority = max(self._max_priority, priority)

        for idx in sorted(idxes):
            heapq._siftdown(self._storage, 0, idx)

        for sample in new_samples:
            heapq.heappushpop(self._storage, sample)
