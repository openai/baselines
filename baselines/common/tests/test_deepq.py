import subprocess

from baselines.common.tests.test_common import _assert_eq


def test_custom_cartpole():
    args = ['--no-render', '--max-timesteps', 1000]
    args = list(map(str, args))
    ok = subprocess.call(['python', '-m', 'baselines.deepq.experiments.custom_cartpole'] + args)
    _assert_eq(ok, 0)

def test_cartpole():
    args = ['--max-timesteps', 1000]
    args = list(map(str, args))
    ok = subprocess.call(['python', '-m', 'baselines.deepq.experiments.train_cartpole'] + args)
    _assert_eq(ok, 0)

    ok = subprocess.call(['python', '-m', 'baselines.deepq.experiments.enjoy_cartpole', '--no-render'])
    _assert_eq(ok, 0)
