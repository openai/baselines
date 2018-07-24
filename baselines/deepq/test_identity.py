import random

import tensorflow as tf

from baselines.deepq import DeepQ, models as deepq_models
from baselines.common.identity_env import IdentityEnv


def test_identity():
    """
    test identity function for DeepQ
    """
    env = IdentityEnv(10)
    random.seed(0)

    tf.set_random_seed(0)

    param_noise = False
    model = deepq_models.mlp([32])

    model = DeepQ(
        env=env,
        q_func=model,
        learning_rate=1e-3,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        param_noise=param_noise,
    )
    model.learn(total_timesteps=10000)

    tf.set_random_seed(0)

    n_trials = 1000
    sum_rew = 0
    obs = env.reset()
    for _ in range(n_trials):
        action, _ = model.predict(obs)
        obs, rew, _, _ = env.step(action)
        sum_rew += rew

    assert sum_rew > 0.9 * n_trials


if __name__ == '__main__':
    test_identity()
