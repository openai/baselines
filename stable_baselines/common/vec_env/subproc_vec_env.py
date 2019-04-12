import multiprocessing
from collections import OrderedDict

import gym
import numpy as np

from stable_baselines.common.vec_env import VecEnv, CloudpickleWrapper
from stable_baselines.common.tile_images import tile_images


def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                observation, reward, done, info = env.step(data)
                if done:
                    observation = env.reset()
                remote.send((observation, reward, done, info))
            elif cmd == 'reset':
                observation = env.reset()
                remote.send(observation)
            elif cmd == 'render':
                remote.send(env.render(*data[0], **data[1]))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            elif cmd == 'env_method':
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == 'get_attr':
                remote.send(getattr(env, data))
            elif cmd == 'set_attr':
                remote.send(setattr(env, data[0], data[1]))
            else:
                raise NotImplementedError
        except EOFError:
            break


class SubprocVecEnv(VecEnv):
    """
    Creates a multiprocess vectorized wrapper for multiple environments

    .. warning::

        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow
        sessions or other non thread-safe libraries are used in the parent (see issue #217).
        However, compared to 'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods,
        users must wrap the code in an ``if __name__ == "__main__":``
        For more information, see the multiprocessing documentation.

    :param env_fns: ([Gym Environment]) Environments to run in subprocesses
    :param start_method: (str) method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'fork' on available platforms, and 'spawn' otherwise.
    """

    def __init__(self, env_fns, start_method=None):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            fork_available = 'fork' in multiprocessing.get_all_start_methods()
            start_method = 'fork' if fork_available else 'spawn'
        ctx = multiprocessing.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return _flatten_obs(obs, self.observation_space)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True

    def render(self, mode='human', *args, **kwargs):
        for pipe in self.remotes:
            # gather images from subprocesses
            # `mode` will be taken into account later
            pipe.send(('render', (args, {'mode': 'rgb_array', **kwargs})))
        imgs = [pipe.recv() for pipe in self.remotes]
        # Create a big image by tiling images from subprocesses
        bigimg = tile_images(imgs)
        if mode == 'human':
            import cv2
            cv2.imshow('vecenv', bigimg[:, :, ::-1])
            cv2.waitKey(1)
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        for pipe in self.remotes:
            pipe.send(('render', {"mode": 'rgb_array'}))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def get_attr(self, attr_name, indices=None):
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('get_attr', attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('set_attr', (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('env_method', (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def _get_target_remotes(self, indices):
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: (None,int,Iterable) refers to indices of envs.
        :return: ([multiprocessing.Connection]) Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]


def _flatten_obs(obs, space):
    """
    Flatten observations, depending on the observation space.

    :param obs: (list<X> or tuple<X> where X is dict<ndarray>, tuple<ndarray> or ndarray) observations.
                A list or tuple of observations, one per environment.
                Each environment observation may be a NumPy array, or a dict or tuple of NumPy arrays.
    :return (OrderedDict<ndarray>, tuple<ndarray> or ndarray) flattened observations.
            A flattened NumPy array or an OrderedDict or tuple of flattened numpy arrays.
            Each NumPy array has the environment index as its first axis.
    """
    assert isinstance(obs, (list, tuple)), "expected list or tuple of observations per environment"
    assert len(obs) > 0, "need observations from at least one environment"

    if isinstance(space, gym.spaces.Dict):
        assert isinstance(space.spaces, OrderedDict), "Dict space must have ordered subspaces"
        assert isinstance(obs[0], dict), "non-dict observation for environment with Dict observation space"
        return OrderedDict([(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()])
    elif isinstance(space, gym.spaces.Tuple):
        assert isinstance(obs[0], tuple), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple((np.stack([o[i] for o in obs]) for i in range(obs_len)))
    else:
        return np.stack(obs)
