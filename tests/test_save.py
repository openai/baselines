import os

import pytest

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


@pytest.mark.slow
@pytest.mark.parametrize("model_class", MODEL_LIST)
def test_model_manipulation(model_class):
    """
    Test if the algorithm (with a given policy) can be loaded and saved without any issues, the environment switching
    works and that the action prediction works

    :param model_class: (BaseRLModel) A RL model
    """

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
            obs, reward, _, _ = env.step(action)
            acc_reward += reward
        acc_reward = sum(acc_reward) / N_TRIALS

        # saving
        model.save("./test_model")

        del model, env

        # loading
        model = model_class.load("./test_model")

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
        if os.path.exists("./test_model"):
            os.remove("./test_model")
