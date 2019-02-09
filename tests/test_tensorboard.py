import os
import shutil

import pytest

from stable_baselines import A2C, ACER, ACKTR, DQN, DDPG, PPO1, PPO2, SAC, TRPO

TENSORBOARD_DIR = '/tmp/tb_dir/'

if os.path.isdir(TENSORBOARD_DIR):
    shutil.rmtree(TENSORBOARD_DIR)

MODEL_DICT = {
    'a2c': (A2C, 'CartPole-v1'),
    'acer': (ACER, 'CartPole-v1'),
    'acktr': (ACKTR, 'CartPole-v1'),
    'dqn': (DQN, 'CartPole-v1'),
    'ddpg': (DDPG, 'Pendulum-v0'),
    'ppo1': (PPO1, 'CartPole-v1'),
    'ppo2': (PPO2, 'CartPole-v1'),
    'sac': (SAC, 'Pendulum-v0'),
    'trpo': (TRPO, 'CartPole-v1'),
}

N_STEPS = 1000


@pytest.mark.parametrize("model_name", MODEL_DICT.keys())
def test_tensorboard(model_name):
    algo, env_id = MODEL_DICT[model_name]
    model = algo('MlpPolicy', env_id, verbose=1, tensorboard_log=TENSORBOARD_DIR)
    model.learn(N_STEPS)
    model.learn(N_STEPS, reset_num_timesteps=False)
