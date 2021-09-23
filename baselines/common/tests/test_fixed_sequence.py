import pytest
from baselines.common.tests.envs.fixed_sequence_env import FixedSequenceEnv

from baselines.common.tests.util import simple_test
from baselines.run import get_learn_function
from baselines.common.tests import mark_slow


common_kwargs = dict(
    seed=0,
    total_timesteps=50000,
)

learn_kwargs = {
    'a2c': {},
    'ppo2': dict(nsteps=10, ent_coef=0.0, nminibatches=1),
    # TODO enable sequential models for trpo_mpi (proper handling of nbatch and nsteps)
    # github issue: https://github.com/openai/baselines/issues/188
    # 'trpo_mpi': lambda e, p: trpo_mpi.learn(policy_fn=p(env=e), env=e, max_timesteps=30000, timesteps_per_batch=100, cg_iters=10, gamma=0.9, lam=1.0, max_kl=0.001)
}


alg_list = learn_kwargs.keys()
rnn_list = ['lstm']

@mark_slow
@pytest.mark.parametrize("alg", alg_list)
@pytest.mark.parametrize("rnn", rnn_list)
def test_fixed_sequence(alg, rnn):
    '''
    Test if the algorithm (with a given policy)
    can learn an identity transformation (i.e. return observation as an action)
    '''

    kwargs = learn_kwargs[alg]
    kwargs.update(common_kwargs)

    env_fn = lambda: FixedSequenceEnv(n_actions=10, episode_len=5)
    learn = lambda e: get_learn_function(alg)(
        env=e,
        network=rnn,
        **kwargs
    )

    simple_test(env_fn, learn, 0.7)


if __name__ == '__main__':
    test_fixed_sequence('ppo2', 'lstm')



