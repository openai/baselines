import pytest

import tensorflow as tf

from baselines import bench, logger
from baselines.deepq import DeepQ, wrap_atari_dqn, models as deepq_models
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
    """
    clears the Tensorflow session, this is needed for sequential testing of the baselines
    """
    tf.reset_default_graph()


@pytest.mark.slow
@pytest.mark.parametrize("policy", ['cnn', 'lstm', 'lnlstm'])
def test_a2c(policy):
    """
    test A2C on atari

    :param policy: (str) the policy to test for A2C
    """
    clear_tf_session()
    a2c_atari.train(env_id=ENV_ID, num_timesteps=NUM_TIMESTEPS, seed=SEED,
                    policy=policy, lr_schedule='constant', num_env=NUM_CPU)


@pytest.mark.slow
@pytest.mark.parametrize("policy", ['cnn', 'lstm'])
def test_acer(policy):
    """
    test ACER on atari

    :param policy: (str) the policy to test for ACER
    """
    clear_tf_session()
    acer_atari.train(env_id=ENV_ID, num_timesteps=NUM_TIMESTEPS, seed=SEED,
                     policy=policy, lr_schedule='constant', num_cpu=NUM_CPU)


@pytest.mark.slow
def test_acktr():
    """
    test ACKTR on atari
    """
    clear_tf_session()
    acktr_atari.train(env_id=ENV_ID, num_timesteps=NUM_TIMESTEPS, seed=SEED, num_cpu=NUM_CPU)


@pytest.mark.slow
def test_deepq():
    """
    test DeepQ on atari
    """
    clear_tf_session()
    logger.configure()
    set_global_seeds(SEED)
    env = make_atari(ENV_ID)
    env = bench.Monitor(env, logger.get_dir())
    env = wrap_atari_dqn(env)
    q_func = deepq_models.cnn_to_mlp(convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], hiddens=[256], dueling=True)

    model = DeepQ(env=env, q_func=q_func, learning_rate=1e-4, buffer_size=10000, exploration_fraction=0.1,
                  exploration_final_eps=0.01, train_freq=4, learning_starts=10000, target_network_update_freq=1000,
                  gamma=0.99, prioritized_replay=True, prioritized_replay_alpha=0.6, checkpoint_freq=10000)
    model.learn(total_timesteps=NUM_TIMESTEPS)

    env.close()


@pytest.mark.slow
def test_ppo1():
    """
    test PPO1 on atari
    """
    clear_tf_session()
    ppo1_atari.train(env_id=ENV_ID, num_timesteps=NUM_TIMESTEPS, seed=SEED)


@pytest.mark.slow
@pytest.mark.parametrize("policy", ['cnn', 'lstm', 'lnlstm', 'mlp'])
def test_ppo2(policy):
    """
    test PPO2 on atari

    :param policy: (str) the policy to test for PPO2
    """
    clear_tf_session()
    ppo2_atari.train(env_id=ENV_ID, num_timesteps=NUM_TIMESTEPS, seed=SEED, policy=policy)


@pytest.mark.slow
def test_trpo():
    """
    test TRPO on atari
    """
    clear_tf_session()
    trpo_atari.train(env_id=ENV_ID, num_timesteps=NUM_TIMESTEPS, seed=SEED)
