import pytest

from baselines.a2c import A2C
from baselines.ppo1 import PPO1
from baselines.ppo2 import PPO2
from baselines.trpo_mpi import TRPO
from baselines.common.identity_env import IdentityEnvMultiBinary, IdentityEnvMultiDiscrete
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.policies import MlpPolicy

MODEL_FUNC_LIST = [
    lambda e: A2C(policy=MlpPolicy, env=e),
    lambda e: PPO1(policy=MlpPolicy, env=e),
    lambda e: PPO2(policy=MlpPolicy, env=e),
    lambda e: TRPO(policy=MlpPolicy, env=e),
]


@pytest.mark.slow
@pytest.mark.parametrize("model_func", MODEL_FUNC_LIST)
def test_identity_multidiscrete(model_func):
    """
    Test if the algorithm (with a given policy)
    can learn an identity transformation (i.e. return observation as an action)
    with a multidiscrete action space

    :param model_func: (lambda (Gym Environment): BaseRLModel) the model generator
    """
    env = DummyVecEnv([lambda: IdentityEnvMultiDiscrete(10)])

    model = model_func(env)
    model.learn(total_timesteps=1000, seed=0)

    n_trials = 1000
    reward_sum = 0
    obs = env.reset()
    for _ in range(n_trials):
        action, _ = model.predict(obs)
        obs, reward, _, _ = env.step(action)
        reward_sum += reward


@pytest.mark.slow
@pytest.mark.parametrize("model_func", MODEL_FUNC_LIST)
def test_identity_multibinary(model_func):
    """
    Test if the algorithm (with a given policy)
    can learn an identity transformation (i.e. return observation as an action)
    with a multibinary action space

    :param model_func: (lambda (Gym Environment): BaseRLModel) the model generator
    """
    env = DummyVecEnv([lambda: IdentityEnvMultiBinary(10)])

    model = model_func(env)
    model.learn(total_timesteps=1000, seed=0)

    n_trials = 1000
    reward_sum = 0
    obs = env.reset()
    for _ in range(n_trials):
        action, _ = model.predict(obs)
        obs, reward, _, _ = env.step(action)
        reward_sum += reward
