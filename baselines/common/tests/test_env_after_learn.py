import pytest
import gym
import tensorflow as tf

from baselines.common.models import cnn
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.run import get_learn_function
from baselines.common.tf_util import make_session

algos = ['a2c', 'acer', 'acktr', 'deepq', 'ppo2', 'trpo_mpi']

@pytest.mark.parametrize('algo', algos)
def test_env_after_learn(algo):
    def make_env():
        env = gym.make('PongNoFrameskip-v4')
        return env

    make_session(make_default=True, graph=tf.Graph())
    env = SubprocVecEnv([make_env])

    learn = get_learn_function(algo)
    network = cnn(one_dim_bias=True)

    # Commenting out the following line resolves the issue, though crash happens at env.reset().
    if algo == 'acktr':
        kwargs = {'is_async': False}
    else:
        kwargs = {}

    learn(network=network, env=env, total_timesteps=0, load_path=None, seed=None, **kwargs)

    env.reset()
    env.close()
