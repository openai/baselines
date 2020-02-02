"""VecEnv implementation using python threads instead of subprocesses."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import threading
import traceback
from baselines.common.vec_env import VecEnv
import numpy as np
from six.moves import queue as Queue  # pylint: disable=redefined-builtin


def thread_worker(send_q, recv_q, env_fn):
    """Similar to SubprocVecEnv.worker(), but for TreadedVecEnv.

    Args:
      send_q: Queue which ThreadedVecEnv sends commands to.
      recv_q: Queue which ThreadedVecEnv receives commands from.
      env_fn: Callable that creates an instance of the environment.
    """
    try:
        env = env_fn()
        while True:
            cmd, data = send_q.get()
            if cmd == 'step':
                ob, reward, done, info = env.step(data)
                if done:
                    ob = env.reset()
                recv_q.put((ob, reward, done, info))
            elif cmd == 'reset':
                ob = env.reset()
                recv_q.put(ob)
            elif cmd == 'render':
                recv_q.put(env.render(mode='rgb_array'))
            elif cmd == 'close':
                env.close()
                break
            elif cmd == 'get_spaces':
                recv_q.put((env.observation_space, env.action_space))
            else:
                raise NotImplementedError
    except Exception as e:  # pylint:disable=broad-except
        print('Worker thread raised exception:', e, '\nExiting program...')
        traceback.print_exc()
        traceback.print_stack()
        # Make sure the whole program dies, instead of having a program that
        # keeps running with dead threads.
        os._exit(1)  # pylint:disable=protected-access


class ThreadedVecEnv(VecEnv):
    """Similar to SubprocVecEnv, but uses python threads instead of subprocs.

    SubprocVecEnv, as the name indicates, use sub-processes to parallelize the
    environments. It happens that some codebases the baselines might be
    integrated are not compatible with that:
    - SubprocVecEnv can use the 'fork' multiprocessing mode, but some
      codebases are not fork-safe, leading to crashes or deadlocks. This can
      happen when a codebase uses POSIX-threads, for instance.
    - When SubprocVecEnv can use the 'spawn' multiprocessing mode, but this
      requires a Python interpreter, which is not available when using an
      embedded interpreter.

    This "ThreadedVecEnv" eliminates this issue by using python threads, which
    are compatible with non-fork-safe codebases and don't require a non-embedded
    python interpreter.

    The drawback of python threads is that the python code is still executed
    sequentially because of the GIL. However, many environments do the heavy
    lifting in C++ and release the GIL while doing so, so python threads are not
    often limiting.
    """

    def __init__(self, env_fns, spaces=None):
        """Initializes a ThreadedVecEnv.

        Args:
        - env_fns: iterable of callables: functions that create environments
            to run in python threads.
            IMPORTANT: the environment created must not be thread-hostile (for
            instance, multiple environments created independently must not share
            a global state in an unsafe way, as is the case with ATARI for
            instance).
        - spaces: ignored.
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.send_queues = [Queue.Queue() for _ in range(nenvs)]
        self.recv_queues = [Queue.Queue() for _ in range(nenvs)]
        self.threads = []
        for (send_q, recv_q, env_fn) in zip(
            self.send_queues, self.recv_queues, env_fns):
          thread = threading.Thread(target=thread_worker,
                                    args=(send_q, recv_q, env_fn))
          thread.daemon = True
          thread.start()
          self.threads.append(thread)

        self.send_queues[0].put(('get_spaces', None))
        observation_space, action_space = self.recv_queues[0].get()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        self._assert_not_closed()
        for send_q, action in zip(self.send_queues, actions):
            send_q.put(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = self._receive_all()
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        self._assert_not_closed()
        self._send_all(('reset', None))
        return np.stack(self._receive_all())

    def reset_task(self):
        self._send_all(('reset_task', None))
        return np.stack(self._receive_all())

    def close_extras(self):
        if self.waiting:
            self._receive_all()
        self._send_all(('close', None))
        for thread in self.threads:
            thread.join()

    def get_images(self):
        self._assert_not_closed()
        self._send_all(('render', None))
        return self._receive_all()

    def _send_all(self, item):
        for send_q in self.send_queues:
            send_q.put(item)

    def _receive_all(self):
        return [recv_q.get() for recv_q in self.recv_queues]

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a Threaded after calling close()"
