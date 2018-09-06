import pytest

from stable_baselines.a2c import A2C
from stable_baselines.acer import ACER
from stable_baselines.acktr import ACKTR
from stable_baselines.deepq import DeepQ
from stable_baselines.ddpg import DDPG
from stable_baselines.ppo1 import PPO1
from stable_baselines.ppo2 import PPO2
from stable_baselines.trpo_mpi import TRPO
from stable_baselines.common.identity_env import IdentityEnv, IdentityEnvBox
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.deepq import policies as deepq_models


learn_func_list = [
    lambda e: A2C(policy=MlpPolicy, learning_rate=1e-3, n_steps=1,
                  gamma=0.7, env=e).learn(total_timesteps=10000, seed=0),
    lambda e: ACER(policy=MlpPolicy, env=e,
                   n_steps=1, replay_ratio=1).learn(total_timesteps=15000, seed=0),
    lambda e: ACKTR(policy=MlpPolicy, env=e, learning_rate=5e-4, n_steps=1).learn(total_timesteps=20000, seed=0),
    lambda e: DeepQ(policy=deepq_models.mlp([32]), batch_size=16, gamma=0.1,
                    exploration_fraction=0.001, env=e).learn(total_timesteps=40000, seed=0),
    lambda e: PPO1(policy=MlpPolicy, env=e, lam=0.5,
                   optim_batchsize=16, optim_stepsize=1e-3).learn(total_timesteps=15000, seed=0),
    lambda e: PPO2(policy=MlpPolicy, env=e, learning_rate=1.5e-3,
                   lam=0.8).learn(total_timesteps=20000, seed=0),
    lambda e: TRPO(policy=MlpPolicy, env=e, max_kl=0.05, lam=0.7).learn(total_timesteps=10000, seed=0),
]


@pytest.mark.slow
@pytest.mark.parametrize("learn_func", learn_func_list)
def test_identity(learn_func):
    """
    Test if the algorithm (with a given policy)
    can learn an identity transformation (i.e. return observation as an action)

    :param learn_func: (lambda (Gym Environment): A2CPolicy) the policy generator
    """
    env = DummyVecEnv([lambda: IdentityEnv(10)])

    model = learn_func(env)

    n_trials = 1000
    reward_sum = 0
    obs = env.reset()
    for _ in range(n_trials):
        action, _ = model.predict(obs)
        obs, reward, _, _ = env.step(action)
        reward_sum += reward
    assert reward_sum > 0.9 * n_trials
    # Free memory
    del model, env


@pytest.mark.slow
def test_identity_ddpg():
    """
    Test if the algorithm (with a given policy)
    can learn an identity transformation (i.e. return observation as an action)
    """
    env = DummyVecEnv([lambda: IdentityEnvBox(eps=0.5)])

    # FIXME: this test fail for now
    model = DDPG(MlpPolicy, env).learn(total_timesteps=10000, seed=0)

    n_trials = 1000
    reward_sum = 0
    obs = env.reset()
    for _ in range(n_trials):
        action, _ = model.predict(obs)
        obs, reward, _, _ = env.step(action)
        reward_sum += reward
    assert reward_sum > 0.9 * n_trials
    # Free memory
    del model, env
