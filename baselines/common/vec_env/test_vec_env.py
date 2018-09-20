"""
Tests for asynchronous vectorized environments.
"""

import gym
import numpy as np
import pytest
from .dummy_vec_env import DummyVecEnv
from .shmem_vec_env import ShmemVecEnv
from .subproc_vec_env import SubprocVecEnv


def assert_envs_equal(env1, env2, num_steps):
    """
    Compare two environments over num_steps steps and make sure
    that the observations produced by each are the same when given
    the same actions.
    """
    assert env1.num_envs == env2.num_envs
    assert env1.action_space.shape == env2.action_space.shape
    assert env1.action_space.dtype == env2.action_space.dtype
    joint_shape = (env1.num_envs,) + env1.action_space.shape

    try:
        obs1, obs2 = env1.reset(), env2.reset()
        assert np.array(obs1).shape == np.array(obs2).shape
        assert np.array(obs1).shape == joint_shape
        assert np.allclose(obs1, obs2)
        np.random.seed(1337)
        for _ in range(num_steps):
            actions = np.array(np.random.randint(0, 0x100, size=joint_shape),
                               dtype=env1.action_space.dtype)
            for env in [env1, env2]:
                env.step_async(actions)
            outs1 = env1.step_wait()
            outs2 = env2.step_wait()
            for out1, out2 in zip(outs1[:3], outs2[:3]):
                assert np.array(out1).shape == np.array(out2).shape
                assert np.allclose(out1, out2)
            assert list(outs1[3]) == list(outs2[3])
    finally:
        env1.close()
        env2.close()


@pytest.mark.parametrize('klass', (ShmemVecEnv, SubprocVecEnv))
@pytest.mark.parametrize('dtype', ('uint8', 'float32'))
def test_vec_env(klass, dtype):  # pylint: disable=R0914
    """
    Test that a vectorized environment is equivalent to
    DummyVecEnv, since DummyVecEnv is less likely to be
    error prone.
    """
    num_envs = 3
    num_steps = 100
    shape = (3, 8)

    def make_fn(seed):
        """
        Get an environment constructor with a seed.
        """
        return lambda: SimpleEnv(seed, shape, dtype)
    fns = [make_fn(i) for i in range(num_envs)]
    env1 = DummyVecEnv(fns)
    env2 = klass(fns)
    assert_envs_equal(env1, env2, num_steps=num_steps)


class SimpleEnv(gym.Env):
    """
    An environment with a pre-determined observation space
    and RNG seed.
    """

    def __init__(self, seed, shape, dtype):
        np.random.seed(seed)
        self._dtype = dtype
        self._start_obs = np.array(np.random.randint(0, 0x100, size=shape),
                                   dtype=dtype)
        self._max_steps = seed + 1
        self._cur_obs = None
        self._cur_step = 0
        # this is 0xFF instead of 0x100 because the Box space includes
        # the high end, while randint does not
        self.action_space = gym.spaces.Box(low=0, high=0xFF, shape=shape, dtype=dtype)
        self.observation_space = self.action_space

    def step(self, action):
        self._cur_obs += np.array(action, dtype=self._dtype)
        self._cur_step += 1
        done = self._cur_step >= self._max_steps
        reward = self._cur_step / self._max_steps
        return self._cur_obs, reward, done, {'foo': 'bar' + str(reward)}

    def reset(self):
        self._cur_obs = self._start_obs
        self._cur_step = 0
        return self._cur_obs

    def render(self, mode=None):
        raise NotImplementedError
