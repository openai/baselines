import numpy as np
import threading


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


class ReplayBuffer(object):
    def __init__(self, buffer_shapes, size_in_transitions, T, sample_transitions):
        """Creates a replay buffer.

        Args:
            buffer_shapes (dict of ints): the shape for all buffers
            that are used in the replay buffer
            size_in_transitions (int): the size of the buffer,
            measured in transitions
            T (int): the time horizon for episodes
            sample_transitions (function): a function that samples
            from the replay buffer
        """
        self.buffer_shapes = buffer_shapes
        self.size = size_in_transitions // T
        self.T = T
        self.sample_transitions = sample_transitions

        # self.buffers is {key: array(size_in_episodes x T or T+1 x dim_key)}
        self.buffers = {key: np.empty([self.size, *shape])
                        for key, shape in buffer_shapes.items()}

        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0

        self.lock = threading.Lock()

    @property
    def full(self):
        with self.lock:
            return self.current_size == self.size

    def sample(self, batch_size):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """
        buffers = {}

        with self.lock:
            assert self.current_size > 0
            for key in self.buffers.keys():
                buffers[key] = self.buffers[key][:self.current_size]

        buffers['o_2'] = buffers['o'][:, 1:, :]
        buffers['ag_2'] = buffers['ag'][:, 1:, :]

        transitions = self.sample_transitions(buffers, batch_size)

        for key in (['r', 'o_2', 'ag_2'] + list(self.buffers.keys())):
            assert key in transitions, "key %s missing from transitions" % key

        return transitions

    def store_episode(self, episode_batch):
        """episode_batch: array(batch_size x (T or T+1) x dim_key)
        """
        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        batch_size = batch_sizes[0]

        with self.lock:
            idxs = self._get_storage_idx(batch_size)

            # load inputs into buffers
            for key in self.buffers.keys():
                self.buffers[key][idxs] = episode_batch[key]

            self.n_transitions_stored += batch_size * self.T

    def get_current_episode_size(self):
        with self.lock:
            return self.current_size

    def get_current_size(self):
        with self.lock:
            return self.current_size * self.T

    def get_transitions_stored(self):
        with self.lock:
            return self.n_transitions_stored

    def clear_buffer(self):
        with self.lock:
            self.current_size = 0

    def _get_storage_idx(self, inc=None):
        inc = inc or 1   # size increment
        assert inc <= self.size, "Batch committed to replay is too large!"
        # go consecutively until you hit the end, and then go randomly.
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)

        # update replay size
        self.current_size = min(self.size, self.current_size+inc)

        if inc == 1:
            idx = idx[0]
        return idx


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
