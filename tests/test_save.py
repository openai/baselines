import os
from io import BytesIO

import pytest
import numpy as np

from stable_baselines import A2C, ACER, ACKTR, DQN, PPO1, PPO2, TRPO
from stable_baselines.common import set_global_seeds
from stable_baselines.common.identity_env import IdentityEnv
from stable_baselines.common.vec_env import DummyVecEnv

N_TRIALS = 2000

MODEL_LIST = [
    A2C,
    ACER,
    ACKTR,
    DQN,
    PPO1,
    PPO2,
    TRPO,
]

STORE_METHODS = [
    "path",
    "file-like"
]


@pytest.mark.slow
@pytest.mark.parametrize("model_class", MODEL_LIST)
@pytest.mark.parametrize("storage_method", STORE_METHODS)
def test_model_manipulation(request, model_class, storage_method):
    """
    Test if the algorithm (with a given policy) can be loaded and saved without any issues, the environment switching
    works and that the action prediction works

    :param model_class: (BaseRLModel) A RL model
    """

    model_fname = './test_model_{}.pkl'.format(request.node.name)

    try:
        env = DummyVecEnv([lambda: IdentityEnv(10)])

        # create and train
        model = model_class(policy="MlpPolicy", env=env)
        model.learn(total_timesteps=50000, seed=0)

        # predict and measure the acc reward
        acc_reward = 0
        set_global_seeds(0)
        obs = env.reset()
        for _ in range(N_TRIALS):
            action, _ = model.predict(obs)
            # Test action probability method
            model.action_probability(obs)
            obs, reward, _, _ = env.step(action)
            acc_reward += reward
        acc_reward = sum(acc_reward) / N_TRIALS

        # test action probability for given (obs, action) pair
        env = model.get_env()
        obs = env.reset()
        observations = np.array([obs for _ in range(10)])
        observations = np.squeeze(observations)
        actions = np.array([env.action_space.sample() for _ in range(10)])
        actions_probas = model.action_probability(observations, actions=actions)
        assert actions_probas.shape == (len(actions), 1), actions_probas.shape
        assert actions_probas.min() >= 0, actions_probas.min()
        assert actions_probas.max() <= 1, actions_probas.max()

        # saving
        if storage_method == "path":  # saving to a path
            model.save(model_fname)
        else:  # saving to a file-like object (BytesIO in this case)
            b_io = BytesIO()
            model.save(b_io)
            model_bytes = b_io.getvalue()
            b_io.close()

        del model, env

        # loading
        if storage_method == "path":  # loading from path
            model = model_class.load(model_fname)
        else:
            b_io = BytesIO(model_bytes)  # loading from file-like object (BytesIO in this case)
            model = model_class.load(b_io)
            b_io.close()

        # changing environment (note: this can be done at loading)
        env = DummyVecEnv([lambda: IdentityEnv(10)])
        model.set_env(env)

        # predict the same output before saving
        loaded_acc_reward = 0
        set_global_seeds(0)
        obs = env.reset()
        for _ in range(N_TRIALS):
            action, _ = model.predict(obs)
            obs, reward, _, _ = env.step(action)
            loaded_acc_reward += reward
        loaded_acc_reward = sum(loaded_acc_reward) / N_TRIALS
        assert abs(acc_reward - loaded_acc_reward) < 0.1, "Error: the prediction seems to have changed between " \
                                                          "loading and saving"

        # learn post loading
        model.learn(total_timesteps=100, seed=0)

        # validate no reset post learning
        loaded_acc_reward = 0
        set_global_seeds(0)
        obs = env.reset()
        for _ in range(N_TRIALS):
            action, _ = model.predict(obs)
            obs, reward, _, _ = env.step(action)
            loaded_acc_reward += reward
        loaded_acc_reward = sum(loaded_acc_reward) / N_TRIALS
        assert abs(acc_reward - loaded_acc_reward) < 0.1, "Error: the prediction seems to have changed between " \
                                                          "pre learning and post learning"

        # predict new values
        obs = env.reset()
        for _ in range(N_TRIALS):
            action, _ = model.predict(obs)
            obs, _, _, _ = env.step(action)

        del model, env

    finally:
        if os.path.exists(model_fname):
            os.remove(model_fname)
