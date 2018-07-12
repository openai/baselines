import random

import tensorflow as tf

from baselines import deepq
from baselines.common.identity_env import IdentityEnv


def test_identity():
    """
    test identity function for DeepQ
    """
    env = IdentityEnv(10)
    random.seed(0)

    tf.set_random_seed(0)

    param_noise = False
    model = deepq.models.mlp([32])

    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=10000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        param_noise=param_noise,
    )

    tf.set_random_seed(0)

    n_trials = 1000
    sum_rew = 0
    obs = env.reset()
    for i in range(n_trials):
        obs, rew, done, _ = env.step(act([obs]))
        sum_rew += rew

    assert sum_rew > 0.9 * n_trials


if __name__ == '__main__':
    test_identity()
