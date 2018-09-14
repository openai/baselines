import pytest

from stable_baselines import A2C, ACER, ACKTR, DeepQ, DDPG, PPO1, PPO2, TRPO
from stable_baselines.ddpg import AdaptiveParamNoiseSpec
from stable_baselines.common.identity_env import IdentityEnv, IdentityEnvBox
from stable_baselines.common.vec_env import DummyVecEnv


# Hyperparameters for learning identity for each RL model
LEARN_FUNC_DICT = {
    'a2c': lambda e: A2C(policy="MlpPolicy", env=e).learn(total_timesteps=1000),
    'acer': lambda e: ACER(policy="MlpPolicy", env=e).learn(total_timesteps=1000),
    'acktr': lambda e: ACKTR(policy="MlpPolicy", env=e).learn(total_timesteps=1000),
    'deepq': lambda e: DeepQ(policy="MlpPolicy", env=e).learn(total_timesteps=1000),
    'ppo1': lambda e: PPO1(policy="MlpPolicy", env=e).learn(total_timesteps=1000),
    'ppo2': lambda e: PPO2(policy="MlpPolicy", env=e).learn(total_timesteps=1000),
    'trpo': lambda e: TRPO(policy="MlpPolicy", env=e).learn(total_timesteps=1000),
}


@pytest.mark.slow
@pytest.mark.parametrize("model_name", ['a2c', 'acer', 'acktr', 'deepq', 'ppo1', 'ppo2', 'trpo'])
def test_identity(model_name):
    """
    Test if the algorithm (with a given policy)
    can learn an identity transformation (i.e. return observation as an action)

    :param model_name: (str) Name of the RL model
    """
    env = DummyVecEnv([lambda: IdentityEnv(10)])

    model = LEARN_FUNC_DICT[model_name](env)

    n_trials = 1000
    obs = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    for _ in range(n_trials):
        assert action == model.predict(obs, deterministic=True)[0]
    # Free memory
    del model, env


@pytest.mark.slow
def test_identity_ddpg():
    """
    Test if the algorithm (with a given policy)
    can learn an identity transformation (i.e. return observation as an action)
    """
    env = DummyVecEnv([lambda: IdentityEnvBox(eps=0.5)])

    std = 0.2
    param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(std), desired_action_stddev=float(std))

    model = DDPG("MlpPolicy", env, param_noise=param_noise, memory_limit=int(1e6))
    model.learn(total_timesteps=1000)

    n_trials = 1000
    obs = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    for _ in range(n_trials):
        assert action == model.predict(obs, deterministic=True)[0]
    # Free memory
    del model, env
