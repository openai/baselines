import numpy as np

from stable_baselines.common.math_util import discount_with_boundaries


def test_discount_with_boundaries():
    """
    test the discount_with_boundaries function
    """
    gamma = 0.9
    rewards = np.array([1.0, 2.0, 3.0, 4.0], 'float32')
    episode_starts = [1.0, 0.0, 0.0, 1.0]
    discounted_rewards = discount_with_boundaries(rewards, episode_starts, gamma)
    assert np.allclose(discounted_rewards, [1 + gamma * 2 + gamma ** 2 * 3, 2 + gamma * 3, 3, 4])
    return
