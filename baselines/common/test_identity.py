import pytest
import tensorflow as tf
import random
import numpy as np
from gym.spaces import np_random

from baselines.a2c import a2c
from baselines.ppo2 import ppo2
from baselines.common.identity_env import IdentityEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.ppo2.policies import MlpPolicy


learn_func_list = [
    lambda e: a2c.learn(policy=MlpPolicy, env=e, seed=0, total_timesteps=50000),
    lambda e: ppo2.learn(policy=MlpPolicy, env=e, total_timesteps=50000, lr=1e-3, nsteps=128, ent_coef=0.01)
]


@pytest.mark.slow
@pytest.mark.parametrize("learn_func", learn_func_list)
def test_identity(learn_func):
    '''
    Test if the algorithm (with a given policy) 
    can learn an identity transformation (i.e. return observation as an action)
    '''
    np.random.seed(0)
    np_random.seed(0)
    random.seed(0)

    env = DummyVecEnv([lambda: IdentityEnv(10)])

    with tf.Graph().as_default(), tf.Session().as_default():
        tf.set_random_seed(0)
        model = learn_func(env)

        N_TRIALS = 1000
        sum_rew = 0
        obs = env.reset()
        for i in range(N_TRIALS):
            obs, rew, done, _ = env.step(model.step(obs)[0])
            sum_rew += rew

        assert sum_rew > 0.9 * N_TRIALS
