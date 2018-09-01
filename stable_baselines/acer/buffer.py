import numpy as np


class Buffer(object):
    def __init__(self, env, n_steps, size=50000):
        """
        A buffer for observations, actions, rewards, mu's, states, masks and dones values

        :param env: (Gym environment) The environment to learn from
        :param n_steps: (int) The number of steps to run for each environment
        :param size: (int) The buffer size in number of steps
        """
        self.n_env = env.num_envs
        self.n_steps = n_steps
        self.n_batch = self.n_env * self.n_steps
        # Each loc contains n_env * n_steps frames, thus total buffer is n_env * size frames
        self.size = size // self.n_steps

        if len(env.observation_space.shape) > 1:
            self.raw_pixels = True
            self.height, self.width, self.n_channels = env.observation_space.shape
            self.obs_dtype = np.uint8
        else:
            self.raw_pixels = False
            if len(env.observation_space.shape) == 1:
                self.obs_dim = env.observation_space.shape[-1]
            else:
                self.obs_dim = 1
            self.obs_dtype = np.float32

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
        """
        Check to see if the buffer has at least the asked number of frames
        
        :param frames: (int) The number of frames checked
        :return: (bool) number of frames in buffer >= number asked
        """
        # Frames per env, so total (n_env * frames) Frames needed
        # Each buffer loc has n_env * n_steps frames
        return self.num_in_buffer >= (frames // self.n_steps)

    def can_sample(self):
        """
        Check if the buffer has at least one frame
        
        :return: (bool) if the buffer has at least one frame
        """
        return self.num_in_buffer > 0

    def decode(self, enc_obs):
        """
        Get the stacked frames of an observation
        
        :param enc_obs: ([float]) the encoded observation
        :return: ([float]) the decoded observation
        """
        # enc_obs has shape [n_envs, n_steps + 1, nh, nw, nc]
        # dones has shape [n_envs, n_steps, nh, nw, nc]
        # returns stacked obs of shape [n_env, (n_steps + 1), nh, nw, nc]
        n_env, n_steps = self.n_env, self.n_steps
        if self.raw_pixels:
            obs_dim = [self.height, self.width, self.n_channels]
        else:
            obs_dim = [self.obs_dim]

        obs = np.zeros([1, n_steps + 1, n_env] + obs_dim, dtype=self.obs_dtype)
        # [n_steps + nstack, n_env, nh, nw, nc]
        x_var = np.reshape(enc_obs, [n_env, n_steps + 1] + obs_dim).swapaxes(1, 0)
        obs[-1, :] = x_var

        if self.raw_pixels:
            obs = obs.transpose((2, 1, 3, 4, 0, 5))
        else:
            obs = obs.transpose((2, 1, 3, 0))
        return np.reshape(obs, [n_env, (n_steps + 1)] + obs_dim[:-1] + [obs_dim[-1]])

    def put(self, enc_obs, actions, rewards, mus, dones, masks):
        """
        Adds a frame to the buffer
        
        :param enc_obs: ([float]) the encoded observation
        :param actions: ([float]) the actions
        :param rewards: ([float]) the rewards
        :param mus: ([float]) the policy probability for the actions
        :param dones: ([bool])
        :param masks: ([bool])
        """
        # enc_obs [n_env, (n_steps + n_stack), nh, nw, nc]
        # actions, rewards, dones [n_env, n_steps]
        # mus [n_env, n_steps, n_act]

        if self.enc_obs is None:
            self.enc_obs = np.empty([self.size] + list(enc_obs.shape), dtype=self.obs_dtype)
            self.actions = np.empty([self.size] + list(actions.shape), dtype=np.int32)
            self.rewards = np.empty([self.size] + list(rewards.shape), dtype=np.float32)
            self.mus = np.empty([self.size] + list(mus.shape), dtype=np.float32)
            self.dones = np.empty([self.size] + list(dones.shape), dtype=np.bool)
            self.masks = np.empty([self.size] + list(masks.shape), dtype=np.bool)

        self.enc_obs[self.next_idx] = enc_obs
        self.actions[self.next_idx] = actions
        self.rewards[self.next_idx] = rewards
        self.mus[self.next_idx] = mus
        self.dones[self.next_idx] = dones
        self.masks[self.next_idx] = masks

        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

    def take(self, arr, idx, envx):
        """
        Reads a frame from a list and index for the asked environment ids
        
        :param arr: (np.ndarray) the array that is read
        :param idx: ([int]) the idx that are read
        :param envx: ([int]) the idx for the environments
        :return: ([float]) the askes frames from the list
        """
        n_env = self.n_env
        out = np.empty([n_env] + list(arr.shape[2:]), dtype=arr.dtype)
        for i in range(n_env):
            out[i] = arr[idx[i], envx[i]]
        return out

    def get(self):
        """
        randomly read a frame from the buffer
        
        :return: ([float], [float], [float], [float], [bool], [float])
                 observations, actions, rewards, mus, dones, maskes
        """
        # returns
        # obs [n_env, (n_steps + 1), nh, nw, n_stack*nc]
        # actions, rewards, dones [n_env, n_steps]
        # mus [n_env, n_steps, n_act]
        n_env = self.n_env
        assert self.can_sample()

        # Sample exactly one id per env. If you sample across envs, then higher correlation in samples from same env.
        idx = np.random.randint(0, self.num_in_buffer, n_env)
        envx = np.arange(n_env)

        dones = self.take(self.dones, idx, envx)
        enc_obs = self.take(self.enc_obs, idx, envx)
        obs = self.decode(enc_obs)
        actions = self.take(self.actions, idx, envx)
        rewards = self.take(self.rewards, idx, envx)
        mus = self.take(self.mus, idx, envx)
        masks = self.take(self.masks, idx, envx)
        return obs, actions, rewards, mus, dones, masks
