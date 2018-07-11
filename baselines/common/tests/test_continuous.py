import subprocess

from baselines.common.tests.test_common import _assert_eq

ENV_ID = 'Pendulum-v0'


def test_ddpg():
    args = ['--env-id', ENV_ID, '--nb-epochs', 2, '--nb-epoch-cycles', 2, '--nb-rollout-steps', 100]
    args = list(map(str, args))
    ok = subprocess.call(['python', '-m', 'baselines.ddpg.main'] + args)
    _assert_eq(ok, 0)
