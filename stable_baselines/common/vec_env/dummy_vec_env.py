from collections import OrderedDict

import numpy as np
from gym import spaces

from . import VecEnv


class DummyVecEnv(VecEnv):
    """
    Creates a simple vectorized wrapper for multiple environments

    :param env_fns: ([Gym Environment]) the list of environments to vectorize
    """
    
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        shapes, dtypes = {}, {}
        self.keys = []
        obs_space = env.observation_space

        if isinstance(obs_space, spaces.Dict):
            assert isinstance(obs_space.spaces, OrderedDict)
            subspaces = obs_space.spaces
        else:
            subspaces = {None: obs_space}

        for key, box in subspaces.items():
            shapes[key] = box.shape
            dtypes[key] = box.dtype
            self.keys.append(key)

        self.buf_obs = {k: np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k]) for k in self.keys}
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        for env_idx in range(self.num_envs):
            obs, self.buf_rews[env_idx], self.buf_dones[env_idx], self.buf_infos[env_idx] =\
                self.envs[env_idx].step(self.actions[env_idx])
            if self.buf_dones[env_idx]:
                obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return (np.copy(self._obs_from_buf()), np.copy(self.buf_rews), np.copy(self.buf_dones),
                self.buf_infos.copy())

    def reset(self):
        for env_idx in range(self.num_envs):
            obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return np.copy(self._obs_from_buf())

    def close(self):
        return

    def get_images(self):
        return [env.render(mode='rgb_array') for env in self.envs]

    def render(self, *args, **kwargs):
        if self.num_envs == 1:
            return self.envs[0].render(*args, **kwargs)
        else:
            return super().render(*args, **kwargs)

    def _save_obs(self, env_idx, obs):
        for key in self.keys:
            if key is None:
                self.buf_obs[key][env_idx] = obs
            else:
                self.buf_obs[key][env_idx] = obs[key]

    def _obs_from_buf(self):
        if self.keys == [None]:
            return self.buf_obs[None]
        else:
            return self.buf_obs

    def env_method(self, method_name, *method_args, **method_kwargs):
        """
        Provides an interface to call arbitrary class methods of vectorized environments

        :param method_name: (str) The name of the env class method to invoke
        :param method_args: (tuple) Any positional arguments to provide in the call
        :param method_kwargs: (dict) Any keyword arguments to provide in the call
        :return: (list) List of items retured by the environment's method call
        """
        return [getattr(env_i, method_name)(*method_args, **method_kwargs) for env_i in self.envs]

    def get_attr(self, attr_name):
        """
        Provides a mechanism for getting class attribues from vectorized environments

        :param attr_name: (str) The name of the attribute whose value to return
        :return: (list) List of values of 'attr_name' in all environments
        """
        return [getattr(env_i, attr_name) for env_i in self.envs]

    def set_attr(self, attr_name, value, indices=None):
        """
        Provides a mechanism for setting arbitrary class attributes inside vectorized environments

        :param attr_name: (str) Name of attribute to assign new value
        :param value: (obj) Value to assign to 'attr_name'
        :param indices: (list,int) Indices of envs to assign value
        :return: (list) in case env access methods might return something, they will be returned in a list
        """
        if indices is None:
            indices = range(len(self.envs))
        elif isinstance(indices, int):
            indices = [indices]
        return [setattr(env_i, attr_name, value) for env_i in [self.envs[i] for i in indices]]
