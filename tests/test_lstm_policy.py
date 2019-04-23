import os

import pytest

from stable_baselines import A2C, ACER, ACKTR, PPO2
from stable_baselines.common.policies import MlpLstmPolicy, LstmPolicy


class CustomLSTMPolicy1(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=128, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse, net_arch=[8, 'lstm', 8],
                         layer_norm=False, feature_extraction="mlp", **_kwargs)


class CustomLSTMPolicy2(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=64, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                         net_arch=['lstm', 8], layer_norm=True, feature_extraction="mlp", **_kwargs)


class CustomLSTMPolicy3(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=64, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                         net_arch=[8, 'lstm'], layer_norm=False, feature_extraction="mlp", **_kwargs)


class CustomLSTMPolicy4(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=64, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                         net_arch=[8, 'lstm', dict(vf=[5, 10], pi=[10])],
                         layer_norm=True, feature_extraction="mlp", **_kwargs)


N_TRIALS = 100

MODELS = [A2C, ACER, ACKTR, PPO2]
LSTM_POLICIES = [MlpLstmPolicy, CustomLSTMPolicy1, CustomLSTMPolicy2, CustomLSTMPolicy3, CustomLSTMPolicy4]


@pytest.mark.parametrize("model_class", MODELS)
@pytest.mark.parametrize("policy", LSTM_POLICIES)
def test_lstm_policy(request, model_class, policy):
    model_fname = './test_model_{}.pkl'.format(request.node.name)

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
        model.save(model_fname)
        del model, env
        # loading
        _ = model_class.load(model_fname, policy=policy)

    finally:
        if os.path.exists(model_fname):
            os.remove(model_fname)
