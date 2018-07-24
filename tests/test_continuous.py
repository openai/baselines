import subprocess

from .test_common import _assert_eq

ENV_ID = 'Pendulum-v0'


def test_ddpg():
    args = ['--env-id', ENV_ID, '--nb-rollout-steps', 100]
    args = list(map(str, args))
    return_code = subprocess.call(['python', '-m', 'baselines.ddpg.main'] + args)
    _assert_eq(return_code, 0)
