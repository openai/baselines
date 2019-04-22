import shutil

import gym
import numpy as np
import pytest

from stable_baselines import A2C, ACER, ACKTR, GAIL, DDPG, DQN, PPO1, PPO2, TRPO, SAC
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.gail import ExpertDataset, generate_expert_traj

EXPERT_PATH_PENDULUM = "stable_baselines/gail/dataset/expert_pendulum.npz"
EXPERT_PATH_DISCRETE = "stable_baselines/gail/dataset/expert_cartpole.npz"


@pytest.mark.parametrize("expert_env", [('Pendulum-v0', EXPERT_PATH_PENDULUM, True),
                                        ('CartPole-v1', EXPERT_PATH_DISCRETE, False)])
def test_gail(expert_env):
    env_id, expert_path, load_from_memory = expert_env
    env = gym.make(env_id)

    traj_data = None
    if load_from_memory:
        traj_data = np.load(expert_path)
        expert_path = None
    dataset = ExpertDataset(traj_data=traj_data, expert_path=expert_path, traj_limitation=10,
                            sequential_preprocessing=True)

    # Note: train for 1M steps to have a working policy
    model = GAIL('MlpPolicy', env, adversary_entcoeff=0.0, lam=0.92, max_kl=0.001,
                 expert_dataset=dataset, hidden_size_adversary=64, verbose=0)

    model.learn(1000)
    model.save("GAIL-{}".format(env_id))
    model = model.load("GAIL-{}".format(env_id), env=env)
    model.learn(1000)

    obs = env.reset()

    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()
    del dataset, model

@pytest.mark.parametrize("generate_env", [
                                            (SAC, 'MlpPolicy', 'Pendulum-v0', 1, 10),
                                            (DQN, 'MlpPolicy', 'CartPole-v1', 1, 10),
                                            (A2C, 'MlpLstmPolicy', 'Pendulum-v0', 1, 10),
                                            (A2C, 'MlpLstmPolicy', 'CartPole-v1', 1, 10),
                                            (A2C, 'CnnPolicy', 'BreakoutNoFrameskip-v4', 8, 1),
                                          ])
def test_generate(generate_env):
    model, policy, env_name, n_env, n_episodes = generate_env

    if n_env > 1:
        env = make_atari_env(env_name, num_env=n_env, seed=0)
        model = model(policy, env, verbose=0)
    else:
        model = model(policy, env_name, verbose=0)

    dataset = generate_expert_traj(model, 'expert', n_timesteps=1000, n_episodes=n_episodes,
                                   image_folder='test_recorded_images')

    assert set(dataset.keys()).issuperset(['actions', 'obs', 'rewards', 'episode_returns', 'episode_starts'])
    assert sum(dataset['episode_starts']) == n_episodes
    assert len(dataset['episode_returns']) == n_episodes
    n_timesteps = len(dataset['episode_starts'])
    for key, val in dataset.items():
        if key != 'episode_returns':
            assert val.shape[0] == n_timesteps, "inconsistent number of timesteps at '{}'".format(key)

    dataset_loaded = np.load('expert.npz')
    assert dataset.keys() == dataset_loaded.keys()
    for key in dataset.keys():
        assert (dataset[key] == dataset_loaded[key]).all(), "different data at '{}'".format(key)


def test_generate_callable():
    """
    Test generating expert trajectories with a callable.
    """
    env = gym.make("CartPole-v1")
    # Here the expert is a random agent
    def dummy_expert(_obs):
        return env.action_space.sample()
    generate_expert_traj(dummy_expert, 'dummy_expert_cartpole', env, n_timesteps=0, n_episodes=10)


def test_pretrain_images():
    env = make_atari_env("PongNoFrameskip-v4", num_env=1, seed=0)
    env = VecFrameStack(env, n_stack=4)
    model = PPO2('CnnPolicy', env)
    generate_expert_traj(model, 'expert_pong', n_timesteps=0, n_episodes=1,
                         image_folder='pretrain_recorded_images')

    expert_path = 'expert_pong.npz'
    dataset = ExpertDataset(expert_path=expert_path, traj_limitation=1, batch_size=32,
                            sequential_preprocessing=True)
    model.pretrain(dataset, n_epochs=2)

    shutil.rmtree('pretrain_recorded_images')
    env.close()
    del dataset, model, env


@pytest.mark.parametrize("model_class", [A2C, GAIL, DDPG, PPO1, PPO2, SAC, TRPO])
def test_behavior_cloning_box(model_class):
    """
    Behavior cloning with continuous actions.
    """
    dataset = ExpertDataset(expert_path=EXPERT_PATH_PENDULUM, traj_limitation=10,
                            sequential_preprocessing=True, verbose=0)
    model = model_class("MlpPolicy", "Pendulum-v0")
    model.pretrain(dataset, n_epochs=20)
    model.save("test-pretrain")
    del dataset, model


@pytest.mark.parametrize("model_class", [A2C, ACER, ACKTR, DQN, GAIL, PPO1, PPO2, TRPO])
def test_behavior_cloning_discrete(model_class):
    dataset = ExpertDataset(expert_path=EXPERT_PATH_DISCRETE, traj_limitation=10,
                            sequential_preprocessing=True, verbose=0)
    model = model_class("MlpPolicy", "CartPole-v1")
    model.pretrain(dataset, n_epochs=10)
    model.save("test-pretrain")
    del dataset, model


def test_dataset_param_validation():
    with pytest.raises(ValueError):
        ExpertDataset()

    traj_data = np.load(EXPERT_PATH_PENDULUM)
    with pytest.raises(ValueError):
        ExpertDataset(traj_data=traj_data, expert_path=EXPERT_PATH_PENDULUM)
