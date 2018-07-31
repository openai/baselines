import subprocess
from functools import partial
import os

import pytest
import gym

from baselines.a2c import A2C
from baselines.acer import ACER
from baselines.acktr import ACKTR
from baselines.ddpg import DDPG
from baselines.ppo1 import PPO1
from baselines.ppo2 import PPO2
from baselines.trpo_mpi import TRPO
from baselines.common import set_global_seeds
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.policies import MlpPolicy
from baselines.ddpg.models import ActorMLP, CriticMLP
from baselines.ppo1.mlp_policy import MlpPolicy as PPO1MlpPolicy
from tests.test_common import _assert_eq

ENV_ID = 'Pendulum-v0'
N_TRIALS = 1000

MODEL_POLICY_LIST = [
    (A2C, {"policy": MlpPolicy}),
    #(ACER, {"policy": MlpPolicy}),
    #(ACKTR, {"policy": MlpPolicy}),
    (DDPG, {"critic_policy": CriticMLP,
            "actor_policy": ActorMLP}),
    (PPO1, {"policy": partial(PPO1MlpPolicy, hid_size=32, num_hid_layers=1)}),
    (PPO2, {"policy": MlpPolicy}),
    (TRPO, {"policy": partial(PPO1MlpPolicy, hid_size=32, num_hid_layers=1)})
]


@pytest.mark.slow
@pytest.mark.parametrize("model_policy", MODEL_POLICY_LIST)
def test_model_manipulation(model_policy):
    """
    Test if the algorithm (with a given policy) can be loaded and saved without any issues, the environment switching
    works and that the action prediction works

    :param model_policy: (BaseRLModel, {Object}) A model, policy pair
    """
    model_class, policies = model_policy

    try:
        env = gym.make(ENV_ID)
        env = DummyVecEnv([lambda: env])

        # create and train
        model = model_class(env=env, **policies)
        model.learn(total_timesteps=5000)

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
        env = gym.make(ENV_ID)
        env = DummyVecEnv([lambda: env])
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
        # assert <5% diff
        assert abs(acc_reward - loaded_acc_reward) / max(acc_reward, loaded_acc_reward) < 0.05, \
            "Error: the prediction seems to have changed between loading and saving"

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
        # assert <5% diff
        assert abs(acc_reward - loaded_acc_reward) / max(acc_reward, loaded_acc_reward) < 0.05, \
            "Error: the prediction seems to have changed between pre learning and post learning"

        # predict new values
        obs = env.reset()
        for _ in range(N_TRIALS):
            action, _ = model.predict(obs)
            obs, _, _, _ = env.step(action)

    finally:
        if os.path.exists("./test_model"):
            os.remove("./test_model")


# def test_ddpg():
#     args = ['--env-id', ENV_ID, '--nb-rollout-steps', 100]
#     args = list(map(str, args))
#     return_code = subprocess.call(['python', '-m', 'baselines.ddpg.main'] + args)
#     _assert_eq(return_code, 0)
