import os

import pytest

from stable_baselines import A2C, ACER, ACKTR, DQN, PPO1, PPO2, TRPO, DDPG
from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines.deepq.policies import FeedForwardPolicy as DQNPolicy
from stable_baselines.ddpg.policies import FeedForwardPolicy as DDPGPolicy

N_TRIALS = 100

class CustomCommonPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomCommonPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[8, dict(vf=[8, 8], pi=[8, 8])],
                                           feature_extraction="mlp")

class CustomDQNPolicy(DQNPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs,
                                           layers=[8, 8],
                                           feature_extraction="mlp")

class CustomDDPGPolicy(DDPGPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDDPGPolicy, self).__init__(*args, **kwargs,
                                           layers=[8, 8],
                                           feature_extraction="mlp")


MODEL_DICT = {
    'a2c': (A2C, CustomCommonPolicy),
    'acer': (ACER, CustomCommonPolicy),
    'acktr': (ACKTR, CustomCommonPolicy),
    'dqn': (DQN, CustomDQNPolicy),
    'ddpg': (DDPG, CustomDDPGPolicy),
    'ppo1': (PPO1, CustomCommonPolicy),
    'ppo2': (PPO2, CustomCommonPolicy),
    'trpo': (TRPO, CustomCommonPolicy),
}


@pytest.mark.parametrize("model_name", MODEL_DICT.keys())
def test_custom_policy(model_name):
    """
    Test if the algorithm (with a custom policy) can be loaded and saved without any issues.
    :param model_class: (BaseRLModel) A RL model
    """

    try:
        model_class, policy = MODEL_DICT[model_name]
        if model_name == 'ddpg':
            env = 'MountainCarContinuous-v0'
        else:
            env = 'CartPole-v1'
        # create and train
        model = model_class(policy, env)
        model.learn(total_timesteps=100, seed=0)

        env = model.get_env()
        # predict and measure the acc reward
        obs = env.reset()
        for _ in range(N_TRIALS):
            action, _ = model.predict(obs)
            # Test action probability method
            if model_name != 'ddpg':
                model.action_probability(obs)
            obs, _, _, _ = env.step(action)
        # saving
        model.save("./test_model")
        del model, env
        # loading
        model = model_class.load("./test_model", policy=policy)

    finally:
        if os.path.exists("./test_model"):
            os.remove("./test_model")
