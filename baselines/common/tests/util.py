import tensorflow as tf
import numpy as np
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

N_TRIALS = 10000
N_EPISODES = 100

def simple_test(env_fn, learn_fn, min_reward_fraction, n_trials=N_TRIALS):
    def seeded_env_fn():
        env = env_fn()
        env.seed(0)
        return env

    np.random.seed(0)
    env = DummyVecEnv([seeded_env_fn])
    with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(allow_soft_placement=True)).as_default():
        tf.set_random_seed(0)
        model = learn_fn(env)
        sum_rew = 0
        done = True
        for i in range(n_trials):
            if done:
                obs = env.reset()
                state = model.initial_state
            if state is not None:
                a, v, state, _ = model.step(obs, S=state, M=[False])
            else:
                a, v, _, _ = model.step(obs)
            obs, rew, done, _ = env.step(a)
            sum_rew += float(rew)
        print("Reward in {} trials is {}".format(n_trials, sum_rew))
        assert sum_rew > min_reward_fraction * n_trials, \
            'sum of rewards {} is less than {} of the total number of trials {}'.format(sum_rew, min_reward_fraction, n_trials)

def reward_per_episode_test(env_fn, learn_fn, min_avg_reward, n_trials=N_EPISODES):
    env = DummyVecEnv([env_fn])
    with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(allow_soft_placement=True)).as_default():
        model = learn_fn(env)
        N_TRIALS = 100
        observations, actions, rewards = rollout(env, model, N_TRIALS)
        rewards = [sum(r) for r in rewards]
        avg_rew = sum(rewards) / N_TRIALS
        print("Average reward in {} episodes is {}".format(n_trials, avg_rew))
        assert avg_rew > min_avg_reward, \
            'average reward in {} episodes ({}) is less than {}'.format(n_trials, avg_rew, min_avg_reward)

def rollout(env, model, n_trials):
    rewards = []
    actions = []
    observations = []
    for i in range(n_trials):
        obs = env.reset()
        state = model.initial_state if hasattr(model, 'initial_state') else None
        episode_rew = []
        episode_actions = []
        episode_obs = []
        while True:
            if state is not None:
                a, v, state, _ = model.step(obs, S=state, M=[False])
            else:
                a,v, _, _ = model.step(obs)

            obs, rew, done, _ = env.step(a)
            episode_rew.append(rew)
            episode_actions.append(a)
            episode_obs.append(obs)
            if done:
                break
        rewards.append(episode_rew)
        actions.append(episode_actions)
        observations.append(episode_obs)
    return observations, actions, rewards

