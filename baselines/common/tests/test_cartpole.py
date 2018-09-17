import pytest
import gym

from baselines.run import get_learn_function
from baselines.common.tests.util import reward_per_episode_test

common_kwargs = dict(
    total_timesteps=30000,
    network='mlp',
    gamma=1.0,
    seed=0,
)

learn_kwargs = {
    'a2c' : dict(nsteps=32, value_network='copy', lr=0.05),
    'acktr': dict(nsteps=32, value_network='copy'),
    'deepq': dict(total_timesteps=20000),
    'ppo2': dict(value_network='copy'),
    'trpo_mpi': {}
}

@pytest.mark.slow
@pytest.mark.parametrize("alg", learn_kwargs.keys())
def test_cartpole(alg):
    '''
    Test if the algorithm (with an mlp policy)
    can learn to balance the cartpole
    '''

    kwargs = common_kwargs.copy()
    kwargs.update(learn_kwargs[alg])

    learn_fn = lambda e: get_learn_function(alg)(env=e, **kwargs)
    def env_fn():

        env = gym.make('CartPole-v0')
        env.seed(0)
        return env

    reward_per_episode_test(env_fn, learn_fn, 100)

if __name__ == '__main__':
    test_cartpole('deepq')
