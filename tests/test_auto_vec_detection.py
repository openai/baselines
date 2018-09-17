import pytest
import numpy as np

from stable_baselines import A2C, ACER, ACKTR, DDPG, DQN, PPO1, PPO2, TRPO
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.identity_env import IdentityEnv, IdentityEnvBox, IdentityEnvMultiBinary, \
    IdentityEnvMultiDiscrete


@pytest.mark.slow
@pytest.mark.parametrize("model_class", [A2C, ACER, ACKTR, DQN, PPO1, PPO2, TRPO])
def test_identity(model_class):
    """
    test the Disrete environment vectorisation detection

    :param model_class: (BaseRLModel) the RL model
    """
    model = model_class(policy="MlpPolicy", env=DummyVecEnv([lambda: IdentityEnv(dim=10)]))

    env0 = IdentityEnv(dim=10)
    env1 = DummyVecEnv([lambda: IdentityEnv(dim=10)])

    n_trials = 100
    for env, expected_shape in [(env0, ()), (env1, (1,))]:
        obs = env.reset()
        for _ in range(n_trials):
            action, _ = model.predict(obs)
            assert np.array(action).shape == expected_shape
            obs, _, _, _ = env.step(action)

    # Free memory
    del model, env


@pytest.mark.slow
@pytest.mark.parametrize("model_class", [A2C, DDPG, PPO1, PPO2, TRPO])
def test_identity_box(model_class):
    """
    test the Box environment vectorisation detection

    :param model_class: (BaseRLModel) the RL model
    """
    model = model_class(policy="MlpPolicy", env=DummyVecEnv([lambda: IdentityEnvBox(eps=0.5)]))

    env0 = IdentityEnvBox()
    env1 = DummyVecEnv([lambda: IdentityEnvBox(eps=0.5)])

    n_trials = 100
    for env, expected_shape in [(env0, (1,)), (env1, (1, 1))]:
        obs = env.reset()
        for _ in range(n_trials):
            action, _ = model.predict(obs)
            assert np.array(action).shape == expected_shape
            obs, _, _, _ = env.step(action)

    # Free memory
    del model, env


@pytest.mark.slow
@pytest.mark.parametrize("model_class", [A2C, PPO1, PPO2, TRPO])
def test_identity_multi_binary(model_class):
    """
    test the MultiBinary environment vectorisation detection

    :param model_class: (BaseRLModel) the RL model
    """
    model = model_class(policy="MlpPolicy", env=DummyVecEnv([lambda: IdentityEnvMultiBinary(dim=10)]))

    env0 = IdentityEnvMultiBinary(dim=10)
    env1 = DummyVecEnv([lambda: IdentityEnvMultiBinary(dim=10)])

    n_trials = 100
    for env, expected_shape in [(env0, (10,)), (env1, (1, 10))]:
        obs = env.reset()
        for _ in range(n_trials):
            action, _ = model.predict(obs)
            assert np.array(action).shape == expected_shape
            obs, _, _, _ = env.step(action)

    # Free memory
    del model, env


@pytest.mark.slow
@pytest.mark.parametrize("model_class", [A2C, PPO1, PPO2, TRPO])
def test_identity_multi_discrete(model_class):
    """
    test the MultiDiscrete environment vectorisation detection

    :param model_class: (BaseRLModel) the RL model
    """
    model = model_class(policy="MlpPolicy", env=DummyVecEnv([lambda: IdentityEnvMultiDiscrete(dim=10)]))

    env0 = IdentityEnvMultiDiscrete(dim=10)
    env1 = DummyVecEnv([lambda: IdentityEnvMultiDiscrete(dim=10)])

    n_trials = 100
    for env, expected_shape in [(env0, (2,)), (env1, (1, 2))]:
        obs = env.reset()
        for _ in range(n_trials):
            action, _ = model.predict(obs)
            assert np.array(action).shape == expected_shape
            obs, _, _, _ = env.step(action)

    # Free memory
    del model, env
