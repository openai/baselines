import heapq
import random
# import threading
import numpy as np
from dstruct.segment_tree import SumSegmentTree, MinSegmentTree


class Buffer(object):
    # gets obs, actions, rewards, mu's, (states, masks), dones
    def __init__(self, env, nsteps, nstack, size=50000):
        self.nenv = env.num_envs
        self.nsteps = nsteps
        self.nh, self.nw, self.nc = env.observation_space.shape
        self.nstack = nstack
        self.nbatch = self.nenv * self.nsteps
        # Each loc contains nenv * nsteps frames, thus total buffer is
        # nenv * size framesps
        self.size = size // (self.nsteps)

        # Memory
        self.enc_obs = None
        self.actions = None
        self.rewards = None
        self.mus = None
        self.dones = None
        self.masks = None

        # Size indexes
        self.next_idx = 0
        self.num_in_buffer = 0

    def has_atleast(self, frames):
        # Frames per env, so total (nenv * frames) Frames needed
        # Each buffer loc has nenv * nsteps frames
        return self.num_in_buffer >= (frames // self.nsteps)

    def can_sample(self):
        return self.num_in_buffer > 0

    # Generate stacked frames
    def decode(self, enc_obs, dones):
        # enc_obs has shape [nenvs, nsteps + nstack, nh, nw, nc]
        # dones has shape [nenvs, nsteps, nh, nw, nc]
        # returns stacked obs of shape [nenv, (nsteps + 1), nh, nw, nstack*nc]
        nstack, nenv, nsteps, nh, nw, nc = self.nstack, self.nenv, \
            self.nsteps, self.nh, self.nw, self.nc
        y = np.empty([nsteps + nstack - 1, nenv, 1, 1, 1], dtype=np.float32)
        obs = np.zeros([nstack, nsteps + nstack, nenv, nh, nw, nc], dtype=np.uint8)
        # [nsteps + nstack, nenv, nh, nw, nc]
        x = np.reshape(
            enc_obs, [nenv, nsteps + nstack, nh, nw, nc]
        ).swapaxes(1, 0)
        y[3:] = np.reshape(
            1.0 - dones, [nenv, nsteps, 1, 1, 1]
        ).swapaxes(1, 0)  # keep
        y[:3] = 1.0
        # y = np.reshape(1 - dones, [nenvs, nsteps, 1, 1, 1])
        for i in range(nstack):
            obs[-(i + 1), i:] = x
            # obs[:,i:,:,:,-(i+1),:] = x
            x = x[:-1] * y
            y = y[1:]
        return np.reshape(obs[:, 3:].transpose((2, 1, 3, 4, 0, 5)), [nenv, (nsteps + 1), nh, nw, nstack * nc])

    def put(self, enc_obs, actions, rewards, mus, dones, masks):
        # enc_obs [nenv, (nsteps + nstack), nh, nw, nc]
        # actions, rewards, dones [nenv, nsteps]
        # mus [nenv, nsteps, nact]

        if self.enc_obs is None:
            self.enc_obs = np.empty(
                [self.size] + list(enc_obs.shape), dtype=np.uint8
            )
            self.actions = np.empty(
                [self.size] + list(actions.shape), dtype=np.int32
            )
            self.rewards = np.empty(
                [self.size] + list(rewards.shape), dtype=np.float32
            )
            self.mus = np.empty(
                [self.size] + list(mus.shape), dtype=np.float32
            )
            self.dones = np.empty(
                [self.size] + list(dones.shape), dtype=np.bool
            )
            self.masks = np.empty(
                [self.size] + list(masks.shape), dtype=np.bool
            )

        self.enc_obs[self.next_idx] = enc_obs
        self.actions[self.next_idx] = actions
        self.rewards[self.next_idx] = rewards
        self.mus[self.next_idx] = mus
        self.dones[self.next_idx] = dones
        self.masks[self.next_idx] = masks

        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

    def take(self, x, idx, envx):
        nenv = self.nenv
        out = np.empty([nenv] + list(x.shape[2:]), dtype=x.dtype)
        for i in range(nenv):
            out[i] = x[idx[i], envx[i]]
        return out

    def get(self):
        # returns
        # obs [nenv, (nsteps + 1), nh, nw, nstack*nc]
        # actions, rewards, dones [nenv, nsteps]
        # mus [nenv, nsteps, nact]
        nenv = self.nenv
        assert self.can_sample()

        # Sample exactly one id per env. If you sample across envs,
        # then higher correlation in samples from same env.
        idx = np.random.randint(0, self.num_in_buffer, nenv)
        envx = np.arange(nenv)

        take = lambda x: self.take(x, idx, envx)  # for i in range(nenv)], axis = 0)
        dones = take(self.dones)
        enc_obs = take(self.enc_obs)
        obs = self.decode(enc_obs, dones)
        actions = take(self.actions)
        rewards = take(self.rewards)
        mus = take(self.mus)
        masks = take(self.masks)
        return obs, actions, rewards, mus, dones, masks


class RingBuffer(object):
    def __init__(self, maxlen, shape, dtype='float32'):
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
        self.data[(self.start + self.length - 1) % self.maxlen] = v


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class Memory(object):
    def __init__(self, limit, action_shape, observation_shape):
        self.limit = limit

        self.observations_actor = RingBuffer(limit, shape=observation_shape)
        self.actions = RingBuffer(limit, shape=action_shape)
        self.rewards = RingBuffer(limit, shape=(1,))
        self.terminals_critic = RingBuffer(limit, shape=(1,))
        self.observations_critic = RingBuffer(limit, shape=observation_shape)

    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.random_integers(self.nb_entries - 2, size=batch_size)

        obs_actor_batch = self.observations_actor.get_batch(batch_idxs)
        obs_critic_batch = self.observations_critic.get_batch(batch_idxs)
        action_batch = self.actions.get_batch(batch_idxs)
        reward_batch = self.rewards.get_batch(batch_idxs)
        terminal_critic_batch = self.terminals_critic.get_batch(batch_idxs)

        result = {
            'obs_actor': array_min2d(obs_actor_batch),
            'obs_critic': array_min2d(obs_critic_batch),
            'rewards': array_min2d(reward_batch),
            'actions': array_min2d(action_batch),
            'terminals_critic': array_min2d(terminal_critic_batch),
        }
        return result

    def append(self, obs_actor, action, reward, obs_critic, terminal_critic,
               training=True):
        if not training:
            return

        self.observations_actor.append(obs_actor)
        self.actions.append(action)
        self.rewards.append(reward)
        self.observations_critic.append(obs_critic)
        self.terminals_critic.append(terminal_critic)

    @property
    def nb_entries(self):
        return len(self.observations_actor)


# class ReplayBuffer(object):
#     def __init__(self, buffer_shapes, size_in_transitions, T, sample_transitions):
#         """Creates a replay buffer.

#         Args:
#             buffer_shapes (dict of ints): the shape for all buffers
#             that are used in the replay buffer
#             size_in_transitions (int): the size of the buffer,
#             measured in transitions
#             T (int): the time horizon for episodes
#             sample_transitions (function): a function that samples
#             from the replay buffer
#         """
#         self.buffer_shapes = buffer_shapes
#         self.size = size_in_transitions // T
#         self.T = T
#         self.sample_transitions = sample_transitions

#         # self.buffers is {key: array(size_in_episodes x T or T+1 x dim_key)}
#         self.buffers = {key: np.empty([self.size, *shape])
#                         for key, shape in buffer_shapes.items()}

#         # memory management
#         self.current_size = 0
#         self.n_transitions_stored = 0

#         self.lock = threading.Lock()

#     @property
#     def full(self):
#         with self.lock:
#             return self.current_size == self.size

#     def sample(self, batch_size):
#         """Returns a dict {key: array(batch_size x shapes[key])}
#         """
#         buffers = {}

#         with self.lock:
#             assert self.current_size > 0
#             for key in self.buffers.keys():
#                 buffers[key] = self.buffers[key][:self.current_size]

#         buffers['o_2'] = buffers['o'][:, 1:, :]
#         buffers['ag_2'] = buffers['ag'][:, 1:, :]

#         transitions = self.sample_transitions(buffers, batch_size)

#         for key in (['r', 'o_2', 'ag_2'] + list(self.buffers.keys())):
#             assert key in transitions, "key %s missing from transitions" % key

#         return transitions

#     def store_episode(self, episode_batch):
#         """episode_batch: array(batch_size x (T or T+1) x dim_key)
#         """
#         batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]
#         assert np.all(np.array(batch_sizes) == batch_sizes[0])
#         batch_size = batch_sizes[0]

#         with self.lock:
#             idxs = self._get_storage_idx(batch_size)

#             # load inputs into buffers
#             for key in self.buffers.keys():
#                 self.buffers[key][idxs] = episode_batch[key]

#             self.n_transitions_stored += batch_size * self.T

#     def get_current_episode_size(self):
#         with self.lock:
#             return self.current_size

#     def get_current_size(self):
#         with self.lock:
#             return self.current_size * self.T

#     def get_transitions_stored(self):
#         with self.lock:
#             return self.n_transitions_stored

#     def clear_buffer(self):
#         with self.lock:
#             self.current_size = 0

#     def _get_storage_idx(self, inc=None):
#         inc = inc or 1   # size increment
#         assert inc <= self.size, "Batch committed to replay is too large!"
#         # go consecutively until you hit the end, and then go randomly.
#         if self.current_size+inc <= self.size:
#             idx = np.arange(self.current_size, self.current_size+inc)
#         elif self.current_size < self.size:
#             overflow = inc - (self.size - self.current_size)
#             idx_a = np.arange(self.current_size, self.size)
#             idx_b = np.random.randint(0, self.current_size, overflow)
#             idx = np.concatenate([idx_a, idx_b])
#         else:
#             idx = np.random.randint(0, self.size, inc)

#         # update replay size
#         self.current_size = min(self.size, self.current_size+inc)

#         if inc == 1:
#             idx = idx[0]
#         return idx


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
        return np.array(obses_t), np.array(actions), np.array(rewards), \
            np.array(obses_tp1), np.array(dones)

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
        idxes = [
            random.randint(0, len(self._storage) - 1) for _ in range(batch_size)
        ]
        return self._encode_sample(idxes)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        """Create Prioritized Replay buffer.

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
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
        #for _ in range(batch_size):
        #    # TODO(szymon): should we ensure no repeats?
        #    mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
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
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
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
        ReplayBuffer.__init__()
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

    def __init__(
            self,
            size,
            alpha,
            n_segs,
            sort_period=10000,
            segs_recompute_thresh=0.01
    ):
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
            idxes = [
                random.randint(0, len(self._storage) - 1) for _ in range(batch_size)
            ]
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
            new_samples.append(
                self.HeapNode(-priority ** self._alpha,
                              self._storage[idx].data)
            )
            self._storage[idx].priority = -np.inf
            self._max_priority = max(self._max_priority, priority)

        for idx in sorted(idxes):
            heapq._siftdown(self._storage, 0, idx)

        for sample in new_samples:
            heapq.heappushpop(self._storage, sample)
