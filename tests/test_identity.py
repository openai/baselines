import pytest
import numpy as np

from stable_baselines import A2C, ACER, ACKTR, DQN, DDPG, PPO1, PPO2, TRPO
from stable_baselines.ddpg import AdaptiveParamNoiseSpec
from stable_baselines.common.identity_env import IdentityEnv, IdentityEnvBox
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common import set_global_seeds


# Hyperparameters for learning identity for each RL model
LEARN_FUNC_DICT = {
    'a2c': lambda e: A2C(policy="MlpPolicy", learning_rate=1e-3, n_steps=1,
                         gamma=0.7, env=e).learn(total_timesteps=10000, seed=0),
    'acer': lambda e: ACER(policy="MlpPolicy", env=e,
                           n_steps=1, replay_ratio=1).learn(total_timesteps=15000, seed=0),
    'acktr': lambda e: ACKTR(policy="MlpPolicy", env=e,
                             learning_rate=5e-4, n_steps=1).learn(total_timesteps=20000, seed=0),
    'dqn': lambda e: DQN(policy="MlpPolicy", batch_size=16, gamma=0.1,
                         exploration_fraction=0.001, env=e).learn(total_timesteps=40000, seed=0),
    'ppo1': lambda e: PPO1(policy="MlpPolicy", env=e, lam=0.5,
                           optim_batchsize=16, optim_stepsize=1e-3).learn(total_timesteps=15000, seed=0),
    'ppo2': lambda e: PPO2(policy="MlpPolicy", env=e,
                           learning_rate=1.5e-3, lam=0.8).learn(total_timesteps=20000, seed=0),
    'trpo': lambda e: TRPO(policy="MlpPolicy", env=e,
                           max_kl=0.05, lam=0.7).learn(total_timesteps=10000, seed=0),
}


@pytest.mark.slow
@pytest.mark.parametrize("model_name", ['a2c', 'acer', 'acktr', 'dqn', 'ppo1', 'ppo2', 'trpo'])
def test_identity(model_name):
    """
    Test if the algorithm (with a given policy)
    can learn an identity transformation (i.e. return observation as an action)

    :param model_name: (str) Name of the RL model
    """
    env = DummyVecEnv([lambda: IdentityEnv(10)])

    model = LEARN_FUNC_DICT[model_name](env)

    n_trials = 1000
    reward_sum = 0
    set_global_seeds(0)
    obs = env.reset()
    for _ in range(n_trials):
        action, _ = model.predict(obs)
        obs, reward, _, _ = env.step(action)
        reward_sum += reward
    assert model.action_probability(obs).shape == (1, 10), "Error: action_probability not returning correct shape"
    assert np.prod(model.action_probability(obs, actions=env.action_space.sample()).shape) == 1, \
        "Error: not scalar probability"
    assert reward_sum > 0.9 * n_trials
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

    model = DDPG("MlpPolicy", env, gamma=0.0, param_noise=param_noise, memory_limit=int(1e6))
    model.learn(total_timesteps=20000, seed=0)

    n_trials = 1000
    reward_sum = 0
    set_global_seeds(0)
    obs = env.reset()
    for _ in range(n_trials):
        action, _ = model.predict(obs)
        obs, reward, _, _ = env.step(action)
        reward_sum += reward
    assert reward_sum > 0.9 * n_trials
    # Free memory
    del model, env
