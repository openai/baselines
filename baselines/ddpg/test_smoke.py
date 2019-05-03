from baselines.common.tests.util import smoketest
def _run(argstr):
    smoketest('--alg=ddpg --env=Pendulum-v0 --num_timesteps=0 ' + argstr)

def test_popart():
    _run('--normalize_returns=True --popart=True')

def test_noise_normal():
    _run('--noise_type=normal_0.1')

def test_noise_ou():
    _run('--noise_type=ou_0.1')

def test_noise_adaptive():
    _run('--noise_type=adaptive-param_0.2,normal_0.1')

