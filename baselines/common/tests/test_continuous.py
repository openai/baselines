import subprocess

ENV_ID = 'Pendulum-v0'

def _assert_eq(left, right):
    assert left == right, '{} != {}'.format(left, right)


def _assert_neq(left, right):
    assert left != right, '{} == {}'.format(left, right)


def test_ddpg():
    args = ['--env-id', ENV_ID, '--nb-epochs', 2, '--nb-epoch-cycles', 2, '--nb-rollout-steps', 100]
    args = list(map(str, args))
    ok = subprocess.call(['python', '-m', 'baselines.ddpg.main'] + args)
    _assert_eq(ok, 0)
