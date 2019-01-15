import os

import gym
import pytest
import tensorflow as tf

from stable_baselines import A2C, ACER, ACKTR, DQN, PPO1, PPO2, TRPO, SAC, DDPG
from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import FeedForwardPolicy as DQNPolicy
from stable_baselines.ddpg.policies import FeedForwardPolicy as DDPGPolicy
from stable_baselines.sac.policies import FeedForwardPolicy as SACPolicy

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

class CustomSACPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self).__init__(*args, **kwargs,
                                           layers=[8, 8],
                                           feature_extraction="mlp")

MODEL_DICT = {
    'a2c': (A2C, CustomCommonPolicy, dict(act_fun=tf.nn.relu)),
    'acer': (ACER, CustomCommonPolicy, dict(act_fun=tf.nn.relu)),
    'acktr': (ACKTR, CustomCommonPolicy, dict(act_fun=tf.nn.relu)),
    'dqn': (DQN, CustomDQNPolicy, dict()),
    'ddpg': (DDPG, CustomDDPGPolicy, dict()),
    'ppo1': (PPO1, CustomCommonPolicy, dict(act_fun=tf.nn.relu)),
    'ppo2': (PPO2, CustomCommonPolicy, dict(act_fun=tf.nn.relu)),
    'sac': (SAC, CustomSACPolicy, dict()),
    'trpo': (TRPO, CustomCommonPolicy, dict(act_fun=tf.nn.relu)),
}


@pytest.mark.parametrize("model_name", MODEL_DICT.keys())
def test_custom_policy(model_name):
    """
    Test if the algorithm (with a custom policy) can be loaded and saved without any issues.
    :param model_class: (BaseRLModel) A RL model
    """

    try:
        model_class, policy, _ = MODEL_DICT[model_name]
        env = 'MountainCarContinuous-v0' if model_name in ['ddpg', 'sac'] else 'CartPole-v1'

        # create and train
        model = model_class(policy, env)
        model.learn(total_timesteps=100, seed=0)

        env = model.get_env()
        # predict and measure the acc reward
        obs = env.reset()
        for _ in range(N_TRIALS):
            action, _ = model.predict(obs)
            # Test action probability method
            if model_name not in ['ddpg', 'sac']:
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


@pytest.mark.parametrize("model_name", MODEL_DICT.keys())
def test_custom_policy_kwargs(model_name):
    """
    Test if the algorithm (with a custom policy) can be loaded and saved without any issues.
    :param model_class: (BaseRLModel) A RL model
    """

    try:
        model_class, policy, policy_kwargs = MODEL_DICT[model_name]
        env = 'MountainCarContinuous-v0' if model_name in ['ddpg', 'sac'] else 'CartPole-v1'

        # create and train
        model = model_class(policy, env, policy_kwargs=policy_kwargs)
        model.learn(total_timesteps=100, seed=0)

        model.save("./test_model")
        del model

        # loading

        env = DummyVecEnv([lambda: gym.make(env)])

        # Load with specifying policy_kwargs
        model = model_class.load("./test_model", policy=policy, env=env, policy_kwargs=policy_kwargs)
        model.learn(total_timesteps=100, seed=0)
        del model

        # Load without specifying policy_kwargs
        model = model_class.load("./test_model", policy=policy, env=env)
        model.learn(total_timesteps=100, seed=0)
        del model

        # Load wit different wrong policy_kwargs
        with pytest.raises(ValueError):
            model = model_class.load("./test_model", policy=policy, env=env, policy_kwargs=dict(wrong="kwargs"))

    finally:
        if os.path.exists("./test_model"):
            os.remove("./test_model")
