"""
Tests for asynchronous vectorized environments.
"""

import gym
import pytest
import os
import glob
import tempfile

from .dummy_vec_env import DummyVecEnv
from .shmem_vec_env import ShmemVecEnv
from .subproc_vec_env import SubprocVecEnv
from .vec_video_recorder import VecVideoRecorder

@pytest.mark.parametrize('klass', (DummyVecEnv, ShmemVecEnv, SubprocVecEnv))
@pytest.mark.parametrize('num_envs', (1, 4))
@pytest.mark.parametrize('video_length', (10, 100))
@pytest.mark.parametrize('video_interval', (1, 50))
def test_video_recorder(klass, num_envs, video_length, video_interval):
    """
    Wrap an existing VecEnv with VevVideoRecorder,
    Make (video_interval + video_length + 1) steps,
    then check that the file is present
    """

    def make_fn():
        env = gym.make('PongNoFrameskip-v4')
        return env
    fns = [make_fn for _ in range(num_envs)]
    env = klass(fns)

    with tempfile.TemporaryDirectory() as video_path:
        env = VecVideoRecorder(env, video_path, record_video_trigger=lambda x: x % video_interval == 0, video_length=video_length)

        env.reset()
        for _ in range(video_interval + video_length + 1):
            env.step([0] * num_envs)
        env.close()


        recorded_video = glob.glob(os.path.join(video_path, "*.mp4"))

        # first and second step
        assert len(recorded_video) == 2
        # Files are not empty
        assert all(os.stat(p).st_size != 0 for p in recorded_video)


