"""
Tests for asynchronous vectorized environments.
"""

import gym
import numpy as np
import pytest
import os
import glob
import tempfile

from .dummy_vec_env import DummyVecEnv
from .shmem_vec_env import ShmemVecEnv
from .subproc_vec_env import SubprocVecEnv
from .test_vec_env import SimpleEnv
from .vec_video_recorder import VecVideoRecorder

@pytest.mark.parametrize('klass', (DummyVecEnv, ShmemVecEnv, SubprocVecEnv))
@pytest.mark.parametrize('num_envs', (1, 4, 16))
@pytest.mark.parametrize('video_length', (10, 100, 500))
@pytest.mark.parametrize('video_interval', (1, 10, 50))
def test_video_recorder(klass, num_envs, video_length, video_interval):
    """
    """
    dtype = 'float32'

    def make_fn():
        env = gym.make('PongNoFrameskip-v4')
        return env
    fns = [make_fn for _ in range(num_envs)]
    env = klass(fns)

    with tempfile.TemporaryDirectory() as video_path:
        env = VecVideoRecorder(env, video_path, video_callable=lambda x: x % video_interval == 0, video_length=video_length)

        env.reset()
        for _ in range(video_interval + video_length + 1):
            env.step([0] * num_envs)
        env.close()


        recorded_video = glob.glob(os.path.join(video_path, "*.mp4"))
        assert len(recorded_video) == 2


