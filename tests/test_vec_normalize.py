import subprocess

import gym
import numpy as np

from stable_baselines.common.running_mean_std import RunningMeanStd
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from .test_common import _assert_eq

ENV_ID = 'Pendulum-v0'


def test_runningmeanstd():
    """Test RunningMeanStd object"""
    for (x_1, x_2, x_3) in [
        (np.random.randn(3), np.random.randn(4), np.random.randn(5)),
        (np.random.randn(3, 2), np.random.randn(4, 2), np.random.randn(5, 2))]:
        rms = RunningMeanStd(epsilon=0.0, shape=x_1.shape[1:])

        x_cat = np.concatenate([x_1, x_2, x_3], axis=0)
        moments_1 = [x_cat.mean(axis=0), x_cat.var(axis=0)]
        rms.update(x_1)
        rms.update(x_2)
        rms.update(x_3)
        moments_2 = [rms.mean, rms.var]

        assert np.allclose(moments_1, moments_2)


def test_vec_env():
    """Test VecNormalize Object"""

    def make_env():
        return gym.make(ENV_ID)

    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.)
    _, done = env.reset(), [False]
    obs = None
    while not done[0]:
        actions = [env.action_space.sample()]
        obs, _, done, _ = env.step(actions)
    assert np.max(obs) <= 10


def test_mpi_runningmeanstd():
    """Test RunningMeanStd object for MPI"""
    return_code = subprocess.call(['mpirun', '--allow-run-as-root', '-np', '2',
                                   'python', '-m', 'stable_baselines.common.mpi_running_mean_std'])
    _assert_eq(return_code, 0)


def test_mpi_moments():
    """
    test running mean std function
    """
    subprocess.check_call(['mpirun', '--allow-run-as-root', '-np', '3', 'python', '-c',
                           'from stable_baselines.common.mpi_moments '
                           'import _helper_runningmeanstd; _helper_runningmeanstd()'])
