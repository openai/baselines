import pytest

# from baselines.acer import acer_simple as acer
from baselines.common.tests.envs.mnist_env import MnistEnv
from baselines.common.tests.util import simple_test
from baselines.run import get_learn_function
from baselines.common.tests import mark_slow

# TODO investigate a2c and ppo2 failures - is it due to bad hyperparameters for this problem?
# GitHub issue https://github.com/openai/baselines/issues/189
common_kwargs = {
    'seed': 0,
    'network':'cnn',
    'gamma':0.9,
    'pad':'SAME'
}

learn_args = {
    'a2c': dict(total_timesteps=50000),
    'acer': dict(total_timesteps=20000),
    'deepq': dict(total_timesteps=5000),
    'acktr': dict(total_timesteps=30000),
    'ppo2': dict(total_timesteps=50000, lr=1e-3, nsteps=128, ent_coef=0.0),
    'trpo_mpi': dict(total_timesteps=80000, timesteps_per_batch=100, cg_iters=10, lam=1.0, max_kl=0.001)
}


#tests pass, but are too slow on travis. Same algorithms are covered
# by other tests with less compute-hungry nn's and by benchmarks
@pytest.mark.skip
@mark_slow
@pytest.mark.parametrize("alg", learn_args.keys())
def test_mnist(alg):
    '''
    Test if the algorithm can learn to classify MNIST digits.
    Uses CNN policy.
    '''

    learn_kwargs = learn_args[alg]
    learn_kwargs.update(common_kwargs)

    learn = get_learn_function(alg)
    learn_fn = lambda e: learn(env=e, **learn_kwargs)
    env_fn = lambda: MnistEnv(episode_len=100)

    simple_test(env_fn, learn_fn, 0.6)

if __name__ == '__main__':
    test_mnist('acer')
