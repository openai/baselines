import subprocess
import os
import gc

import pytest
import gym

from baselines.a2c import A2C
# TODO: add support for continuous actions
# from baselines.acer import ACER
# from baselines.acktr import ACKTR
from baselines.ddpg import DDPG
from baselines.ppo1 import PPO1
from baselines.ppo2 import PPO2
from baselines.trpo_mpi import TRPO
from baselines.common import set_global_seeds
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.policies import MlpPolicy
from tests.test_common import _assert_eq

ENV_ID = 'Pendulum-v0'
N_TRIALS = 1000

MODEL_LIST = [
    A2C,
    # ACER,
    # ACKTR,
    DDPG,
    PPO1,
    PPO2,
    TRPO
]


@pytest.mark.slow
@pytest.mark.parametrize("model_class", MODEL_LIST)
def test_model_manipulation(model_class):
    """
    Test if the algorithm can be loaded and saved without any issues, the environment switching
    works and that the action prediction works

    :param model_class: (BaseRLModel) A model
    """
    try:
        env = gym.make(ENV_ID)
        env = DummyVecEnv([lambda: env])

        # create and train
        model = model_class(policy=MlpPolicy, env=env)
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

        # Free memory
        del model
        del obs

    finally:
        if os.path.exists("./test_model"):
            os.remove("./test_model")


def test_ddpg():
    # Free memory, otherwise, travis will complain with an out of memory error:
    # OSError: [Errno 12] Cannot allocate memory
    gc.collect()
    gc.collect()
    args = ['--env-id', ENV_ID, '--nb-rollout-steps', 100]
    args = list(map(str, args))
    return_code = subprocess.call(['python', '-m', 'baselines.ddpg.main'] + args)
    _assert_eq(return_code, 0)
