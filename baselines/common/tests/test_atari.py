import pytest
import sys

import tensorflow as tf

from baselines import deepq, bench, logger
from baselines.common import set_global_seeds
from baselines.common.atari_wrappers import make_atari
import baselines.a2c.run_atari as a2c_atari
import baselines.acer.run_atari as acer_atari
import baselines.acktr.run_atari as acktr_atari
import baselines.ppo1.run_atari as ppo1_atari
import baselines.ppo2.run_atari as ppo2_atari
import baselines.trpo_mpi.run_atari as trpo_atari


ENV_ID = 'BreakoutNoFrameskip-v4'
SEED = 3
NUM_TIMESTEPS = 2500
NUM_CPU = 4


def clear_tf_session():
    if tf.get_default_session() is not None:
        print("Session!!! {}".format(tf.get_default_session()), file=sys.stderr)
    tf.reset_default_graph()


@pytest.mark.slow
@pytest.mark.parametrize("policy", ['cnn', 'lstm', 'lnlstm'])
def test_a2c(policy):
    clear_tf_session()
    with tf.Graph().as_default():
        a2c_atari.train(env_id=ENV_ID, num_timesteps=NUM_TIMESTEPS, seed=SEED,
                        policy=policy, lrschedule='constant', num_env=NUM_CPU)


@pytest.mark.slow
@pytest.mark.parametrize("policy", ['cnn', 'lstm'])
def test_acer(policy):
    clear_tf_session()
    with tf.Graph().as_default():
        acer_atari.train(env_id=ENV_ID, num_timesteps=NUM_TIMESTEPS, seed=SEED,
                         policy=policy, lrschedule='constant', num_cpu=NUM_CPU)


@pytest.mark.slow
def test_acktr():
    clear_tf_session()
    with tf.Graph().as_default():
        acktr_atari.train(env_id=ENV_ID, num_timesteps=NUM_TIMESTEPS, seed=SEED, num_cpu=NUM_CPU)


@pytest.mark.slow
def test_deepq():
    clear_tf_session()
    with tf.Graph().as_default():
        logger.configure()
        set_global_seeds(SEED)
        env = make_atari(ENV_ID)
        env = bench.Monitor(env, logger.get_dir())
        env = deepq.wrap_atari_dqn(env)
        model = deepq.models.cnn_to_mlp(
            convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
            hiddens=[256],
            dueling=True,
        )

        with tf.Session():
            deepq.learn(
                env,
                q_func=model,
                lr=1e-4,
                max_timesteps=NUM_TIMESTEPS,
                buffer_size=10000,
                exploration_fraction=0.1,
                exploration_final_eps=0.01,
                train_freq=4,
                learning_starts=10000,
                target_network_update_freq=1000,
                gamma=0.99,
                prioritized_replay=True,
                prioritized_replay_alpha=0.6,
                checkpoint_freq=10000
            )

        env.close()


@pytest.mark.slow
def test_ppo1():
    clear_tf_session()
    with tf.Graph().as_default():
        ppo1_atari.train(env_id=ENV_ID, num_timesteps=NUM_TIMESTEPS, seed=SEED)


@pytest.mark.slow
@pytest.mark.parametrize("policy", ['cnn', 'lstm', 'lnlstm', 'mlp'])
def test_ppo2(policy):
    clear_tf_session()
    with tf.Graph().as_default():
        ppo2_atari.train(env_id=ENV_ID, num_timesteps=NUM_TIMESTEPS, seed=SEED, policy=policy)

# FIXME: This is broken as the graph is not correctly removed after PP01 is run.
# FIXME: However the fix requires a refactoring of ALL the models to deal with their session internaly.
# @pytest.mark.slow
# def test_trpo():
#     clear_tf_session()
#     with tf.Graph().as_default():
#         trpo_atari.train(env_id=ENV_ID, num_timesteps=NUM_TIMESTEPS, seed=SEED)
