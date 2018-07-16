import subprocess

from .test_common import _assert_eq


def test_custom_cartpole():
    args = ['--no-render', '--max-timesteps', 1000]
    args = list(map(str, args))
    return_code = subprocess.call(['python', '-m', 'baselines.deepq.experiments.custom_cartpole'] + args)
    _assert_eq(return_code, 0)

def test_cartpole():
    args = ['--max-timesteps', 1000]
    args = list(map(str, args))
    return_code = subprocess.call(['python', '-m', 'baselines.deepq.experiments.train_cartpole'] + args)
    _assert_eq(return_code, 0)

    return_code = subprocess.call(['python', '-m', 'baselines.deepq.experiments.enjoy_cartpole', '--no-render'])
    _assert_eq(return_code, 0)

def test_mountaincar():
    args = ['--max-timesteps', 1000]
    args = list(map(str, args))
    return_code = subprocess.call(['python', '-m', 'baselines.deepq.experiments.train_mountaincar'] + args)
    _assert_eq(return_code, 0)

    return_code = subprocess.call(['python', '-m', 'baselines.deepq.experiments.enjoy_mountaincar', '--no-render'])
    _assert_eq(return_code, 0)
