import os

import pytest

from stable_baselines import A2C, ACER, PPO2
from stable_baselines.common.policies import MlpLstmPolicy

N_TRIALS = 100

MODELS = [A2C, ACER, PPO2]
LSTM_POLICIES = [MlpLstmPolicy]


@pytest.mark.parametrize("model_class", MODELS)
@pytest.mark.parametrize("policy", LSTM_POLICIES)
def test_lstm_policy(model_class, policy):
    try:
        # create and train
        if model_class == PPO2:
            model = model_class(policy, 'CartPole-v1', nminibatches=1)
        else:
            model = model_class(policy, 'CartPole-v1')
        model.learn(total_timesteps=100, seed=0)

        env = model.get_env()
        # predict and measure the acc reward
        obs = env.reset()
        for _ in range(N_TRIALS):
            action, _ = model.predict(obs)
            obs, _, _, _ = env.step(action)
        # saving
        model.save("./test_model")
        del model, env
        # loading
        model = model_class.load("./test_model", policy=policy)

    finally:
        if os.path.exists("./test_model"):
            os.remove("./test_model")
