import numpy as np
from itertools import chain
from multiprocessing import Process, Pipe
from . import VecEnv, CloudpickleWrapper


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

def chunk_worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()

    env_fns_chunk = env_fn_wrapper.x

    list_envs = [env_fn() for env_fn in env_fns_chunk]

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                results = []
                for env, d in zip(list_envs, data):
                    ob, reward, done, info = env.step(d)
                    if done:
                        ob = env.reset()
                    results.append([ob, reward, done, info])
                remote.send(results)
            elif cmd == 'reset':
                obs = [env.reset() for env in list_envs]
                remote.send(obs)
            elif cmd == 'render':
                imgs = [env.render('rgb_array') for env in list_envs]
                remote.send(imgs)
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((list_envs[0].observation_space, list_envs[0].action_space))
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()


class SubprocChunkVecEnv(VecEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """
    def __init__(self, env_fns, chunk_size=1, spaces=None):
        """
        Arguments:
        env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)

        self.chunk_size = min(chunk_size, nenvs)
        env_fns_chunks = list(chunks(env_fns, chunk_size))
        num_chunks = len(env_fns_chunks)

        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(num_chunks)])
        self.ps = [Process(target=chunk_worker, args=(work_remote, remote, CloudpickleWrapper(env_fn_chunk)))
                   for (work_remote, remote, env_fn_chunk) in zip(self.work_remotes, self.remotes, env_fns_chunks)]

        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.viewer = None
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        self._assert_not_closed()
        action_chunks = chunks(actions, self.chunk_size)
        for remote, action_chunk in zip(self.remotes, action_chunks):
            remote.send(('step', action_chunk))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        results = list(chain.from_iterable(results))
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        results = chain.from_iterable(results)
        return np.stack(results)

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def get_images(self):
        self._assert_not_closed()
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        imgs = chain.from_iterable(imgs)

        return imgs

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocChunkVecEnv after calling close()"
