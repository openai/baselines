import pytest

from stable_baselines import A2C, ACER, ACKTR, DQN, DDPG, PPO1, PPO2, SAC, TRPO, TD3
from stable_baselines.common.noise import NormalActionNoise

N_STEPS_TRAINING = 5000
SEED = 0

# Weird stuff: TD3 would fail if another algorithm is tested before
# with n_cpu_tf_sess > 1
@pytest.mark.parametrize("algo", [A2C, ACKTR, ACER, DDPG, DQN, PPO1, PPO2, SAC, TRPO, TD3])
def test_deterministic_training_common(algo):
    results = [[], []]
    rewards = [[], []]
    kwargs = {'n_cpu_tf_sess': 1}
    if algo in [DDPG, TD3, SAC]:
        env_id = 'Pendulum-v0'
        kwargs.update({'action_noise': NormalActionNoise(0.0, 0.1)})
    else:
        env_id = 'CartPole-v1'
        if algo == DQN:
            kwargs.update({'learning_starts': 100})

    for i in range(2):
        model = algo('MlpPolicy', env_id, seed=SEED, **kwargs)
        model.learn(N_STEPS_TRAINING)
        env = model.get_env()
        obs = env.reset()
        for _ in range(100):
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, _, _ = env.step(action)
            results[i].append(action)
            rewards[i].append(reward)
    assert sum(results[0]) == sum(results[1]), results
    assert sum(rewards[0]) == sum(rewards[1]), rewards
