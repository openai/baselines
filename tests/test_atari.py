import pytest

from stable_baselines import bench, logger
from stable_baselines.deepq import DQN, wrap_atari_dqn, CnnPolicy
from stable_baselines.common import set_global_seeds
from stable_baselines.common.atari_wrappers import make_atari
import stable_baselines.a2c.run_atari as a2c_atari
import stable_baselines.acer.run_atari as acer_atari
import stable_baselines.acktr.run_atari as acktr_atari
import stable_baselines.ppo1.run_atari as ppo1_atari
import stable_baselines.ppo2.run_atari as ppo2_atari
import stable_baselines.trpo_mpi.run_atari as trpo_atari


ENV_ID = 'BreakoutNoFrameskip-v4'
SEED = 3
NUM_TIMESTEPS = 500
NUM_CPU = 2


@pytest.mark.slow
@pytest.mark.parametrize("policy", ['cnn', 'lstm', 'lnlstm'])
def test_a2c(policy):
    """
    test A2C on atari

    :param policy: (str) the policy to test for A2C
    """
    a2c_atari.train(env_id=ENV_ID, num_timesteps=NUM_TIMESTEPS, seed=SEED,
                    policy=policy, lr_schedule='constant', num_env=NUM_CPU)


@pytest.mark.slow
@pytest.mark.parametrize("policy", ['cnn', 'lstm'])
def test_acer(policy):
    """
    test ACER on atari

    :param policy: (str) the policy to test for ACER
    """
    acer_atari.train(env_id=ENV_ID, num_timesteps=NUM_TIMESTEPS, seed=SEED,
                     policy=policy, lr_schedule='constant', num_cpu=NUM_CPU)


@pytest.mark.slow
def test_acktr():
    """
    test ACKTR on atari
    """
    acktr_atari.train(env_id=ENV_ID, num_timesteps=NUM_TIMESTEPS, seed=SEED, num_cpu=NUM_CPU)


@pytest.mark.slow
def test_deepq():
    """
    test DeepQ on atari
    """
    logger.configure()
    set_global_seeds(SEED)
    env = make_atari(ENV_ID)
    env = bench.Monitor(env, logger.get_dir())
    env = wrap_atari_dqn(env)

    model = DQN(env=env, policy=CnnPolicy, learning_rate=1e-4, buffer_size=10000, exploration_fraction=0.1,
                exploration_final_eps=0.01, train_freq=4, learning_starts=10000, target_network_update_freq=1000,
                gamma=0.99, prioritized_replay=True, prioritized_replay_alpha=0.6, checkpoint_freq=10000)
    model.learn(total_timesteps=NUM_TIMESTEPS)

    env.close()
    del model, env


@pytest.mark.slow
def test_ppo1():
    """
    test PPO1 on atari
    """
    ppo1_atari.train(env_id=ENV_ID, num_timesteps=NUM_TIMESTEPS, seed=SEED)


@pytest.mark.slow
@pytest.mark.parametrize("policy", ['cnn', 'lstm', 'lnlstm', 'mlp'])
def test_ppo2(policy):
    """
    test PPO2 on atari

    :param policy: (str) the policy to test for PPO2
    """
    ppo2_atari.train(env_id=ENV_ID, num_timesteps=NUM_TIMESTEPS, seed=SEED, policy=policy)


@pytest.mark.slow
def test_trpo():
    """
    test TRPO on atari
    """
    trpo_atari.train(env_id=ENV_ID, num_timesteps=NUM_TIMESTEPS, seed=SEED)
