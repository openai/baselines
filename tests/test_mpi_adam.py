import subprocess

from .test_common import _assert_eq


def test_mpi_adam():
    """Test RunningMeanStd object for MPI"""
    return_code = subprocess.call(['mpirun', '--allow-run-as-root', '-np', '2',
                                   'python', '-m', 'stable_baselines.common.mpi_adam'])
    _assert_eq(return_code, 0)


def test_mpi_adam_ppo1():
    """Running test for ppo1"""
    return_code = subprocess.call(['mpirun', '--allow-run-as-root', '-np', '2',
                                   'python', '-m',
                                   'stable_baselines.ppo1.experiments.train_cartpole'])
    _assert_eq(return_code, 0)
