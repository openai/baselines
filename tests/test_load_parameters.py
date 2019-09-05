import os
from io import BytesIO

import pytest
import numpy as np

from stable_baselines import A2C, ACER, ACKTR, DQN, PPO1, PPO2, TRPO
from stable_baselines.common.identity_env import IdentityEnv
from stable_baselines.common.vec_env import DummyVecEnv

MODEL_LIST = [
    A2C,
    ACER,
    ACKTR,
    DQN,
    PPO1,
    PPO2,
    TRPO,
]

@pytest.mark.parametrize("model_class", MODEL_LIST)
def test_load_parameters(request, model_class):
    """
    Test if ``load_parameters`` loads given parameters correctly (the model actually changes)
    and that the backwards compatability with a list of params works

    :param model_class: (BaseRLModel) A RL model
    """
    env = DummyVecEnv([lambda: IdentityEnv(10)])

    # create model
    model = model_class(policy="MlpPolicy", env=env)

    # test action probability for given (obs, action) pair
    env = model.get_env()
    obs = env.reset()
    observations = np.array([obs for _ in range(10)])
    observations = np.squeeze(observations)

    actions = np.array([env.action_space.sample() for _ in range(10)])
    original_actions_probas = model.action_probability(observations, actions=actions)

    # Get dictionary of current parameters
    params = model.get_parameters()
    # Modify all parameters to be random values
    random_params = dict((param_name, np.random.random(size=param.shape)) for param_name, param in params.items())
    # Update model parameters with the new zeroed values
    model.load_parameters(random_params)
    # Get new action probas
    new_actions_probas = model.action_probability(observations, actions=actions)

    # Check that at least some action probabilities are different now
    assert not np.any(np.isclose(original_actions_probas, new_actions_probas)), "Action probabilities did not change " \
                                                                                "after changing model parameters."
    # Also check that new parameters are there (they should be random_params)
    new_params = model.get_parameters()
    comparisons = [np.all(np.isclose(new_params[key], random_params[key])) for key in random_params.keys()]
    assert all(comparisons), "Parameters of model are not the same as provided ones."


    # Now test the backwards compatibility with params being a list instead of a dict.
    # Get the ordering of parameters.
    tf_param_list = model.get_parameter_list()
    # Make random parameters negative to make sure the results should be different from
    # previous random values
    random_param_list = [-np.random.random(size=tf_param.shape) for tf_param in tf_param_list]
    model.load_parameters(random_param_list)

    # Compare results against the previous load
    new_actions_probas_list = model.action_probability(observations, actions=actions)
    assert not np.any(np.isclose(new_actions_probas, new_actions_probas_list)), "Action probabilities did not " \
                                                                                "change after changing model " \
                                                                                "parameters (list)."

    # Test file/file-like object loading for load_parameters.
    # Save whatever is stored in model now, assign random parameters,
    # load parameters from file with load_parameters and check if original probabilities
    # are restored
    original_actions_probas = model.action_probability(observations, actions=actions)
    model_fname = './test_model_{}.zip'.format(request.node.name)

    try:
        # Save model to a file and file-like buffer
        # (partly copy/paste from test_save)
        model.save(model_fname)
        b_io = BytesIO()
        model.save(b_io)
        model_bytes = b_io.getvalue()
        b_io.close()

        random_params = dict((param_name, np.random.random(size=param.shape)) for param_name, param in params.items())
        model.load_parameters(random_params)
        # Previous tests confirm that load_parameters works,
        # so just right into testing loading from file
        model.load_parameters(model_fname)
        new_actions_probas = model.action_probability(observations, actions=actions)
        assert np.all(np.isclose(original_actions_probas, new_actions_probas)), "Action probabilities changed " \
                                                                                "after load_parameters from a file."
        # Reset with random parameters again
        model.load_parameters(random_params)
        # Now load from file-like (copy/paste from test_save)
        b_io = BytesIO(model_bytes)
        model.load_parameters(b_io)
        b_io.close()
        new_actions_probas = model.action_probability(observations, actions=actions)
        assert np.all(np.isclose(original_actions_probas, new_actions_probas)), "Action probabilities changed after" \
                                                                                "load_parameters from a file-like."
    finally:
        if os.path.exists(model_fname):
            os.remove(model_fname)


    # Test `exact_match` functionality of load_parameters
    original_actions_probas = model.action_probability(observations, actions=actions)
    # Create dictionary with one variable name missing
    truncated_random_params = dict((param_name, np.random.random(size=param.shape))
                                   for param_name, param in params.items())
    # Remove some element
    _ = truncated_random_params.pop(list(truncated_random_params.keys())[0])
    # With exact_match=True, this should be an expection
    with pytest.raises(RuntimeError):
        model.load_parameters(truncated_random_params, exact_match=True)
    # Make sure we did not update model regardless
    new_actions_probas = model.action_probability(observations, actions=actions)
    assert np.all(np.isclose(original_actions_probas, new_actions_probas)), "Action probabilities changed " \
                                                                            "after load_parameters raised " \
                                                                            "RunTimeError (exact_match=True)."

    # With False, this should be fine
    model.load_parameters(truncated_random_params, exact_match=False)
    # Also check that results changed, again
    new_actions_probas = model.action_probability(observations, actions=actions)
    assert not np.any(np.isclose(original_actions_probas, new_actions_probas)), "Action probabilities did not " \
                                                                                "change after changing model " \
                                                                                "parameters (exact_match=False)."

    del model, env
