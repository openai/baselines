import os
from io import BytesIO
import json
import zipfile

import pytest
import numpy as np

from stable_baselines import A2C, ACER, ACKTR, DQN, PPO1, PPO2, TRPO
from stable_baselines.common import set_global_seeds
from stable_baselines.common.identity_env import IdentityEnv
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy, FeedForwardPolicy

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

STORE_FORMAT = [
    "zip",
    "cloudpickle"
]

@pytest.mark.slow
@pytest.mark.parametrize("model_class", MODEL_LIST)
@pytest.mark.parametrize("storage_method", STORE_METHODS)
@pytest.mark.parametrize("store_format", STORE_FORMAT)
def test_model_manipulation(request, model_class, storage_method, store_format):
    """
    Test if the algorithm (with a given policy) can be loaded and saved without any issues, the environment switching
    works and that the action prediction works

    :param model_class: (BaseRLModel) A RL model
    :param storage_method: (str) Should file be saved to a file ("path") or to a buffer 
        ("file-like")
    :param store_format: (str) Save format, either "zip" or "cloudpickle".
    """
    
    # Use postfix ".model" so we can remove the file later
    model_fname = './test_model_{}.model'.format(request.node.name)
    store_as_cloudpickle = store_format == "cloudpickle"

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
            model.save(model_fname, cloudpickle=store_as_cloudpickle)
        else:  # saving to a file-like object (BytesIO in this case)
            b_io = BytesIO()
            model.save(b_io, cloudpickle=store_as_cloudpickle)
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

class CustomMlpPolicy(FeedForwardPolicy):
    """A dummy "custom" policy to test out custom_objects"""
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(CustomMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps,
                                              n_batch, reuse, feature_extraction="mlp",
                                              **_kwargs)


@pytest.mark.parametrize("model_class", MODEL_LIST)
def test_save_custom_objects(request, model_class):
    """
    Test feeding custom_objects in model.load(...) function
    """
    # Skip DQN (not an actor-critic policy)
    if model_class == DQN:
        return

    model_fname = './test_model_{}.zip'.format(request.node.name)

    try:
        env = DummyVecEnv([lambda: IdentityEnv(10)])

        # Create and save model with default MLP policy
        model = model_class(policy=MlpPolicy, env=env)
        model.save(model_fname)

        del model, env

        # Corrupt "policy" serialization in the file
        data_file = zipfile.ZipFile(model_fname, "r")
        # Load all data (can't just update one file in the archive)
        parameter_list = data_file.read("parameter_list")
        parameters = data_file.read("parameters")
        class_data = json.loads(data_file.read("data").decode())
        data_file.close()

        # Corrupt serialization of the "policy"
        class_data["policy"][":serialized:"] = (
            "Adding this should break serialization" +
            class_data["policy"][":serialized:"]
        )

        # And dump everything back to the model file
        data_file = zipfile.ZipFile(model_fname, "w")
        data_file.writestr("data", json.dumps(class_data))
        data_file.writestr("parameter_list", parameter_list)
        data_file.writestr("parameters", parameters)
        data_file.close()

        # Try loading the model. This should
        # result in an error
        with pytest.raises(RuntimeError):
            model = model_class.load(model_fname)

        # Load model with custom objects ("custom" MlpPolicy)
        # and it should work fine.
        # Note: We could load model with just vanilla
        #       MlpPolicy, too.
        model = model_class.load(
            model_fname,
            custom_objects={
                "policy": CustomMlpPolicy
            }
        )

        # Make sure we loaded custom MLP policy
        assert model.policy == CustomMlpPolicy
        del model

    finally:
        if os.path.exists(model_fname):
            os.remove(model_fname)
