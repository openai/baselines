import tensorflow as tf
import random

from baselines import deepq
from baselines.common.identity_env import IdentityEnv


def test_identity():

    with tf.Graph().as_default():
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

        N_TRIALS = 1000
        sum_rew = 0
        obs = env.reset()
        for i in range(N_TRIALS):
            obs, rew, done, _ = env.step(act([obs]))
            sum_rew += rew

        assert sum_rew > 0.9 * N_TRIALS


if __name__ == '__main__':
    test_identity()
