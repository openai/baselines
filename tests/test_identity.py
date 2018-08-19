import pytest

from baselines.a2c import A2C
from baselines.acer import ACER
from baselines.acktr import ACKTR
from baselines.deepq import DeepQ
from baselines.ppo1 import PPO1
from baselines.ppo2 import PPO2
from baselines.trpo_mpi import TRPO
from baselines.common.identity_env import IdentityEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.policies import MlpPolicy
from baselines.deepq import models as deepq_models

learn_func_list = [
    lambda e: A2C(policy=MlpPolicy, env=e).learn(total_timesteps=50000, seed=0),
    lambda e: ACER(policy=MlpPolicy, env=e, replay_start=100, lr_schedule='constant',
                   gamma=0.5, delta=10, rprop_alpha=0.999,
                   ent_coef=0.0, learning_rate=1e-3, replay_ratio=10).learn(total_timesteps=150000, seed=0),
    lambda e: ACKTR(policy=MlpPolicy, env=e, learning_rate=5e-4, n_steps=4).learn(total_timesteps=100000, seed=0),
    lambda e: DeepQ(policy=deepq_models.mlp([32]), env=e).learn(total_timesteps=50000, seed=0),
    lambda e: PPO1(policy=MlpPolicy, env=e, optim_stepsize=5e-3).learn(total_timesteps=75000, seed=0),
    lambda e: PPO2(policy=MlpPolicy, env=e, learning_rate=1e-3).learn(total_timesteps=50000, seed=0),
    lambda e: TRPO(policy=MlpPolicy, env=e, max_kl=0.05, lam=0.9,
                   timesteps_per_batch=128, vf_stepsize=1e-1).learn(total_timesteps=50000, seed=0),
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
