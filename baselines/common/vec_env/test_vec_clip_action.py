import gym
import pytest

from .dummy_vec_env import DummyVecEnv
from .shmem_vec_env import ShmemVecEnv
from .subproc_vec_env import SubprocVecEnv
from .vec_clip_action import VecClipAction


@pytest.mark.parametrize('klass', (DummyVecEnv, ShmemVecEnv, SubprocVecEnv))
def test_vec_clip_action(klass):
    def make_fn():
        env = gym.make('MountainCarContinuous-v0')
        return env
    fns = [make_fn for _ in range(2)]
    env = klass(fns)
    env_clipped = VecClipAction(env)

    env.reset()
    env_clipped.reset()

    action = [[0.5], [1000]]

    _, rewards, _, _ = env.step(action)
    _, rewards_clipped, _, _ = env_clipped.step(action)

    assert rewards[0] == rewards_clipped[0]
    assert abs(rewards[1]) > abs(rewards_clipped[1])
