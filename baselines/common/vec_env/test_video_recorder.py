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
@pytest.mark.parametrize('num_envs', (1, 3, 16))
@pytest.mark.parametrize('video_length', (10, 50, 200))
@pytest.mark.parametrize('video_interval', (1, 10, 100))
def test_video_recorder(klass, num_envs, video_length, video_interval):
    """
    """
    num_steps = 100
    shape = (3, 8)
    dtype = 'float32'

    def make_fn():
        env = gym.make('PongNoFrameskip-v4')
        return env
    fns = [make_fn for _ in range(num_envs)]
    env = klass(fns)

    video_path = os.path.join(os.getcwd(), "video")
    os.mkdir(video_path)

    try:
        env = VecVideoRecorder(env, video_path, video_callable=lambda x: x % video_interval == 0, video_length=video_length)

        env.reset()
        for _ in range(video_interval + video_length + 1):
            _ = env.step(np.array(np.random.randint(0, 0x100, size=env.action_space),
                                       dtype=dtype))
        env.close()


        recorded_video = glob.glob("video/*.mp4")
        assert len(recorded_video) == 2
    finally:
        list(map(os.remove, glob.glob("video/*")))
        os.rmdir(video_path)


