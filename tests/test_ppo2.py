import os

import pytest

from stable_baselines import PPO2


@pytest.mark.parametrize("cliprange", [0.2, lambda x: 0.1 * x])
@pytest.mark.parametrize("cliprange_vf", [None, 0.2, lambda x: 0.3 * x, -1.0])
def test_clipping(cliprange, cliprange_vf):
    """Test the different clipping (policy and vf)"""
    model = PPO2('MlpPolicy', 'CartPole-v1',
                 cliprange=cliprange, cliprange_vf=cliprange_vf).learn(1000)
    model.save('./ppo2_clip.zip')
    env = model.get_env()
    model = PPO2.load('./ppo2_clip.zip', env=env)
    model.learn(1000)

    if os.path.exists('./ppo2_clip.zip'):
        os.remove('./ppo2_clip.zip')
