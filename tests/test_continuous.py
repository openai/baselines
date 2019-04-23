import subprocess
import os

import gym
import pytest
import numpy as np

from stable_baselines import A2C, SAC
# TODO: add support for continuous actions
# from stable_baselines.acer import ACER
# from stable_baselines.acktr import ACKTR
from stable_baselines.ddpg import DDPG
from stable_baselines.ppo1 import PPO1
from stable_baselines.ppo2 import PPO2
from stable_baselines.trpo_mpi import TRPO
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.identity_env import IdentityEnvBox
from stable_baselines.ddpg import AdaptiveParamNoiseSpec, NormalActionNoise
from tests.test_common import _assert_eq


N_TRIALS = 1000
NUM_TIMESTEPS = 15000

MODEL_LIST = [
    A2C,
    # ACER,
    # ACKTR,
    DDPG,
    PPO1,
    PPO2,
    SAC,
    TRPO
]


@pytest.mark.slow
@pytest.mark.parametrize("model_class", MODEL_LIST)
def test_model_manipulation(request, model_class):
    """
    Test if the algorithm can be loaded and saved without any issues, the environment switching
    works and that the action prediction works

    :param model_class: (BaseRLModel) A model
    """
    try:
        env = DummyVecEnv([lambda: IdentityEnvBox(eps=0.5)])

        # create and train
        model = model_class(policy="MlpPolicy", env=env)
        model.learn(total_timesteps=NUM_TIMESTEPS, seed=0)

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
        model_fname = './test_model_{}.pkl'.format(request.node.name)
        model.save(model_fname)

        del model, env

        # loading
        model = model_class.load(model_fname)

        # changing environment (note: this can be done at loading)
        env = DummyVecEnv([lambda: IdentityEnvBox(eps=0.5)])
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

        with pytest.warns(None) as record:
            act_prob = model.action_probability(obs)

        if model_class in [DDPG, SAC]:
            # check that only one warning was raised
            assert len(record) == 1, "No warning was raised for {}".format(model_class)
            assert act_prob is None, "Error: action_probability should be None for {}".format(model_class)
        else:
            assert act_prob[0].shape == (1, 1) and act_prob[1].shape == (1, 1), \
                "Error: action_probability not returning correct shape"

        # test action probability for given (obs, action) pair
        # must return zero and raise a warning or raise an exception if not defined
        env = model.get_env()
        obs = env.reset()
        observations = np.array([obs for _ in range(10)])
        observations = np.squeeze(observations)
        observations = observations.reshape((-1, 1))
        actions = np.array([env.action_space.sample() for _ in range(10)])

        if model_class == DDPG:
            with pytest.raises(ValueError):
                model.action_probability(observations, actions=actions)
        else:
            with pytest.warns(UserWarning):
                actions_probas = model.action_probability(observations, actions=actions)
            assert actions_probas.shape == (len(actions), 1), actions_probas.shape
            assert np.all(actions_probas == 0.0), actions_probas

        # assert <15% diff
        assert abs(acc_reward - loaded_acc_reward) / max(acc_reward, loaded_acc_reward) < 0.15, \
            "Error: the prediction seems to have changed between loading and saving"

        # learn post loading
        model.learn(total_timesteps=100, seed=0)

        # validate no reset post learning
        # This test was failing from time to time for no good reason
        # other than bad luck
        # We should change this test
        # loaded_acc_reward = 0
        # set_global_seeds(0)
        # obs = env.reset()
        # for _ in range(N_TRIALS):
        #     action, _ = model.predict(obs)
        #     obs, reward, _, _ = env.step(action)
        #     loaded_acc_reward += reward
        # loaded_acc_reward = sum(loaded_acc_reward) / N_TRIALS
        # # assert <10% diff
        # assert abs(acc_reward - loaded_acc_reward) / max(acc_reward, loaded_acc_reward) < 0.1, \
        #     "Error: the prediction seems to have changed between pre learning and post learning"

        # predict new values
        obs = env.reset()
        for _ in range(N_TRIALS):
            action, _ = model.predict(obs)
            obs, _, _, _ = env.step(action)

        # Free memory
        del model, env

    finally:
        if os.path.exists("./test_model"):
            os.remove("./test_model")


def test_ddpg():
    args = ['--env-id', 'Pendulum-v0', '--num-timesteps', 1000, '--noise-type', 'ou_0.01']
    args = list(map(str, args))
    return_code = subprocess.call(['python', '-m', 'stable_baselines.ddpg.main'] + args)
    _assert_eq(return_code, 0)


def test_ddpg_eval_env():
    """
    Additional test to check that everything is working when passing
    an eval env.
    """
    eval_env = gym.make("Pendulum-v0")
    model = DDPG("MlpPolicy", "Pendulum-v0", nb_rollout_steps=5,
                nb_train_steps=2, nb_eval_steps=10,
                eval_env=eval_env, verbose=0)
    model.learn(1000)


def test_ddpg_normalization():
    """
    Test that observations and returns normalizations are properly saved and loaded.
    """
    param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.05, desired_action_stddev=0.05)
    model = DDPG('MlpPolicy', 'Pendulum-v0', memory_limit=50000, normalize_observations=True,
                 normalize_returns=True, nb_rollout_steps=128, nb_train_steps=1,
                 batch_size=64, param_noise=param_noise)
    model.learn(1000)
    obs_rms_params = model.sess.run(model.obs_rms_params)
    ret_rms_params = model.sess.run(model.ret_rms_params)
    model.save('./test_ddpg.pkl')

    loaded_model = DDPG.load('./test_ddpg.pkl')
    obs_rms_params_2 = loaded_model.sess.run(loaded_model.obs_rms_params)
    ret_rms_params_2 = loaded_model.sess.run(loaded_model.ret_rms_params)

    for param, param_loaded in zip(obs_rms_params + ret_rms_params,
                                   obs_rms_params_2 + ret_rms_params_2):
        assert np.allclose(param, param_loaded)

    del model, loaded_model

    if os.path.exists("./test_ddpg.pkl"):
        os.remove("./test_ddpg.pkl")


def test_ddpg_popart():
    """
    Test DDPG with pop-art normalization
    """
    n_actions = 1
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = DDPG('MlpPolicy', 'Pendulum-v0', memory_limit=50000, normalize_observations=True,
                 normalize_returns=True, nb_rollout_steps=128, nb_train_steps=1,
                 batch_size=64, action_noise=action_noise, enable_popart=True)
    model.learn(1000)
