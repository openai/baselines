import os

import pytest

from baselines.a2c import A2C
from baselines.acer import ACER
from baselines.acktr import ACKTR
from baselines.deepq import DeepQ
from baselines.ppo1 import PPO1
from baselines.ppo2 import PPO2
from baselines.trpo_mpi import TRPO
from baselines.common import set_global_seeds
from baselines.common.identity_env import IdentityEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.policies import MlpPolicy
from baselines.deepq import models as deepq_models

N_TRIALS = 2000

MODEL_POLICY_LIST = [
    (A2C, MlpPolicy),
    (ACER, MlpPolicy),
    (ACKTR, MlpPolicy),
    (DeepQ, deepq_models.mlp([32])),
    (PPO1, MlpPolicy),
    (PPO2, MlpPolicy),
    (TRPO, MlpPolicy)
]


@pytest.mark.slow
@pytest.mark.parametrize("model_policy", MODEL_POLICY_LIST)
def test_model_manipulation(model_policy):
    """
    Test if the algorithm (with a given policy) can be loaded and saved without any issues, the environment switching
    works and that the action prediction works

    :param model_policy: (BaseRLModel, Object) A model, policy pair
    """
    model_class, policy = model_policy

    try:
        env = DummyVecEnv([lambda: IdentityEnv(10)])

        # check the env is deterministic
        action = [env.action_space.sample()]
        set_global_seeds(0)
        obs = env.step(action)[0]
        for _ in range(N_TRIALS):
            set_global_seeds(0)
            assert obs == env.step(action)[0], "Error: environment tested not deterministic with the same seed"

        # create and train
        model = model_class(policy=policy, env=env)
        model.learn(total_timesteps=50000)

        # predict and measure the acc reward
        acc_reward = 0
        obs = env.reset()
        set_global_seeds(0)
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
        obs = env.reset()
        set_global_seeds(0)
        for _ in range(N_TRIALS):
            action, _ = model.predict(obs)
            obs, reward, _, _ = env.step(action)
            loaded_acc_reward += reward
        loaded_acc_reward = sum(loaded_acc_reward) / N_TRIALS
        assert abs(acc_reward - loaded_acc_reward) < 0.1, "Error: the prediction seems to have changed between " \
                                                          "loading and saving"

        # learn post loading
        model.learn(total_timesteps=1000)

        # validate no reset post learning
        loaded_acc_reward = 0
        obs = env.reset()
        set_global_seeds(0)
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
