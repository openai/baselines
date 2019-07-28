import os

import pytest

from stable_baselines import HER, DQN, SAC, DDPG, TD3
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
from stable_baselines.her.replay_buffer import KEY_TO_GOAL_STRATEGY
from stable_baselines.common.bit_flipping_env import BitFlippingEnv
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize

N_BITS = 10


def model_predict(model, env, n_steps, additional_check=None):
    """
    Test helper
    :param model: (rl model)
    :param env: (gym.Env)
    :param n_steps: (int)
    :param additional_check: (callable)
    """
    obs = env.reset()
    for _ in range(n_steps):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)

        if additional_check is not None:
            additional_check(obs, action, reward, done)

        if done:
            obs = env.reset()


@pytest.mark.parametrize('goal_selection_strategy', list(GoalSelectionStrategy))
@pytest.mark.parametrize('model_class', [DQN, SAC, DDPG, TD3])
@pytest.mark.parametrize('discrete_obs_space', [False, True])
def test_her(model_class, goal_selection_strategy, discrete_obs_space):
    env = BitFlippingEnv(N_BITS, continuous=model_class in [DDPG, SAC, TD3],
                         max_steps=N_BITS, discrete_obs_space=discrete_obs_space)

    # Take random actions 10% of the time
    kwargs = {'random_exploration': 0.1} if model_class in [DDPG, SAC, TD3] else {}
    model = HER('MlpPolicy', env, model_class, n_sampled_goal=4, goal_selection_strategy=goal_selection_strategy,
                verbose=0, **kwargs)
    model.learn(1000)


@pytest.mark.parametrize('model_class', [DDPG, SAC, DQN, TD3])
def test_long_episode(model_class):
    """
    Check that the model does not break when the replay buffer is still empty
    after the first rollout (because the episode is not over).
    """
    # n_bits > nb_rollout_steps
    n_bits = 10
    env = BitFlippingEnv(n_bits, continuous=model_class in [DDPG, SAC, TD3],
                         max_steps=n_bits)
    kwargs = {}
    if model_class == DDPG:
        kwargs['nb_rollout_steps'] = 9  # < n_bits
    elif model_class in [DQN, SAC, TD3]:
        kwargs['batch_size'] = 8  # < n_bits
        kwargs['learning_starts'] = 0

    model = HER('MlpPolicy', env, model_class, n_sampled_goal=4, goal_selection_strategy='future',
                verbose=0, **kwargs)
    model.learn(200)


@pytest.mark.parametrize('goal_selection_strategy', [list(KEY_TO_GOAL_STRATEGY.keys())[0]])
@pytest.mark.parametrize('model_class', [DQN, SAC, DDPG, TD3])
def test_model_manipulation(model_class, goal_selection_strategy):
    env = BitFlippingEnv(N_BITS, continuous=model_class in [DDPG, SAC, TD3], max_steps=N_BITS)
    env = DummyVecEnv([lambda: env])

    model = HER('MlpPolicy', env, model_class, n_sampled_goal=3, goal_selection_strategy=goal_selection_strategy,
                verbose=0)
    model.learn(1000)

    model_predict(model, env, n_steps=100, additional_check=None)

    model.save('./test_her')
    del model

    # NOTE: HER does not support VecEnvWrapper yet
    with pytest.raises(AssertionError):
        model = HER.load('./test_her', env=VecNormalize(env))

    model = HER.load('./test_her')

    # Check that the model raises an error when the env
    # is not wrapped (or no env passed to the model)
    with pytest.raises(ValueError):
        model.predict(env.reset())

    env_ = BitFlippingEnv(N_BITS, continuous=model_class in [DDPG, SAC, TD3], max_steps=N_BITS)
    env_ = HERGoalEnvWrapper(env_)

    model_predict(model, env_, n_steps=100, additional_check=None)

    model.set_env(env)
    model.learn(1000)

    model_predict(model, env_, n_steps=100, additional_check=None)

    assert model.n_sampled_goal == 3

    del model

    env = BitFlippingEnv(N_BITS, continuous=model_class in [DDPG, SAC, TD3], max_steps=N_BITS)
    model = HER.load('./test_her', env=env)
    model.learn(1000)

    model_predict(model, env_, n_steps=100, additional_check=None)

    assert model.n_sampled_goal == 3

    if os.path.isfile('./test_her.pkl'):
        os.remove('./test_her.pkl')
