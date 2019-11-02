import subprocess

import gym
import numpy as np

from stable_baselines.common.running_mean_std import RunningMeanStd
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from .test_common import _assert_eq

ENV_ID = 'Pendulum-v0'


def make_env():
    return gym.make(ENV_ID)


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


def check_rms_equal(rmsa, rmsb):
    assert np.all(rmsa.mean == rmsb.mean)
    assert np.all(rmsa.var == rmsb.var)
    assert np.all(rmsa.count == rmsb.count)


def check_vec_norm_equal(norma, normb):
    assert norma.observation_space == normb.observation_space
    assert norma.action_space == normb.action_space
    assert norma.num_envs == normb.num_envs

    check_rms_equal(norma.obs_rms, normb.obs_rms)
    check_rms_equal(norma.ret_rms, normb.ret_rms)
    assert norma.clip_obs == normb.clip_obs
    assert norma.clip_reward == normb.clip_reward
    assert norma.norm_obs == normb.norm_obs
    assert norma.norm_reward == normb.norm_reward

    assert np.all(norma.ret == normb.ret)
    assert norma.gamma == normb.gamma
    assert norma.epsilon == normb.epsilon
    assert norma.training == normb.training


def test_vec_env(tmpdir):
    """Test VecNormalize Object"""
    clip_obs = 0.5
    clip_reward = 5.0

    orig_venv = DummyVecEnv([make_env])
    norm_venv = VecNormalize(orig_venv, norm_obs=True, norm_reward=True, clip_obs=clip_obs, clip_reward=clip_reward)
    _, done = norm_venv.reset(), [False]
    while not done[0]:
        actions = [norm_venv.action_space.sample()]
        obs, rew, done, _ = norm_venv.step(actions)
        assert np.max(np.abs(obs)) <= clip_obs
        assert np.max(np.abs(rew)) <= clip_reward

    path = str(tmpdir.join("vec_normalize"))
    norm_venv.save(path)
    deserialized = VecNormalize.load(path, venv=orig_venv)
    check_vec_norm_equal(norm_venv, deserialized)


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
