import os
import tempfile
import pytest
import tensorflow as tf
import numpy as np

from baselines.common.tests.envs.mnist_env import MnistEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.run import get_learn_function
from baselines.common.tf_util import make_session, get_session

from functools import partial


learn_kwargs = {
    'deepq': {},
    'a2c': {},
    'acktr': {},
    'ppo2': {'nminibatches': 1, 'nsteps': 10},
    'trpo_mpi': {},
}

network_kwargs = {
    'mlp': {},
    'cnn': {'pad': 'SAME'},
    'lstm': {},
    'cnn_lnlstm': {'pad': 'SAME'}
}


@pytest.mark.parametrize("learn_fn", learn_kwargs.keys())
@pytest.mark.parametrize("network_fn", network_kwargs.keys())
def test_serialization(learn_fn, network_fn):
    '''
    Test if the trained model can be serialized
    '''


    if network_fn.endswith('lstm') and learn_fn in ['acktr', 'trpo_mpi', 'deepq']:
            # TODO make acktr work with recurrent policies
            # and test
            # github issue: https://github.com/openai/baselines/issues/194
            return

    env = DummyVecEnv([lambda: MnistEnv(10, episode_len=100)])
    ob = env.reset().copy()
    learn = get_learn_function(learn_fn)

    kwargs = {}
    kwargs.update(network_kwargs[network_fn])
    kwargs.update(learn_kwargs[learn_fn])


    learn = partial(learn, env=env, network=network_fn, seed=0, **kwargs)

    with tempfile.TemporaryDirectory() as td:
        model_path = os.path.join(td, 'serialization_test_model')

        with tf.Graph().as_default(), make_session().as_default():
            model = learn(total_timesteps=100)
            model.save(model_path)
            mean1, std1 = _get_action_stats(model, ob)
            variables_dict1 = _serialize_variables()

        with tf.Graph().as_default(), make_session().as_default():
            model = learn(total_timesteps=0, load_path=model_path)
            mean2, std2 = _get_action_stats(model, ob)
            variables_dict2 = _serialize_variables()

        for k, v in variables_dict1.items():
            np.testing.assert_allclose(v, variables_dict2[k], atol=0.01,
                err_msg='saved and loaded variable {} value mismatch'.format(k))

        np.testing.assert_allclose(mean1, mean2, atol=0.5)
        np.testing.assert_allclose(std1, std2, atol=0.5)



def _serialize_variables():
    sess = get_session()
    variables = tf.trainable_variables()
    values = sess.run(variables)
    return {var.name: value for var, value in zip(variables, values)}


def _get_action_stats(model, ob):
    ntrials = 1000
    if model.initial_state is None or model.initial_state == []:
        actions = np.array([model.step(ob)[0] for _ in range(ntrials)])
    else:
        actions = np.array([model.step(ob, S=model.initial_state, M=[False])[0] for _ in range(ntrials)])

    mean = np.mean(actions, axis=0)
    std = np.std(actions, axis=0)

    return mean, std

