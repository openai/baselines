import itertools
import argparse

import gym
import numpy as np
import tensorflow as tf

import stable_baselines.common.tf_util as tf_utils
from stable_baselines import logger, deepq
from stable_baselines.deepq.replay_buffer import ReplayBuffer
from stable_baselines.deepq.policies import FeedForwardPolicy
from stable_baselines.common.schedules import LinearSchedule


class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(vf=[64], pi=[64])],
                                           feature_extraction="mlp")


def main(args):
    """
    Train a DQN agent on cartpole env
    :param args: (Parsed Arguments) the input arguments
    """
    with tf_utils.make_session(8) as sess:
        # Create the environment
        env = gym.make("CartPole-v0")
        # Create all the functions necessary to train the model
        act, train, update_target, _ = deepq.build_train(
            q_func=CustomPolicy,
            ob_space=env.observation_space,
            ac_space=env.action_space,
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
            sess=sess
        )
        # Create the replay buffer
        replay_buffer = ReplayBuffer(50000)
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        exploration = LinearSchedule(schedule_timesteps=10000, initial_p=1.0, final_p=0.02)

        # Initialize the parameters and copy them to the target network.
        tf_utils.initialize()
        update_target()

        episode_rewards = [0.0]
        obs = env.reset()
        for step in itertools.count():
            # Take action and update exploration to the newest value
            action = act(obs[None], update_eps=exploration.value(step))[0]
            new_obs, rew, done, _ = env.step(action)
            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = env.reset()
                episode_rewards.append(0)

            if len(episode_rewards[-101:-1]) == 0:
                mean_100ep_reward = -np.inf
            else:
                mean_100ep_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

            is_solved = step > 100 and mean_100ep_reward >= 200

            if args.no_render and step > args.max_timesteps:
                break

            if is_solved:
                if args.no_render:
                    break
                # Show off the result
                env.render()
            else:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if step > 1000:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
                    train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                # Update target network periodically.
                if step % 1000 == 0:
                    update_target()

            if done and len(episode_rewards) % 10 == 0:
                logger.record_tabular("steps", step)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular("mean episode reward", mean_100ep_reward)
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(step)))
                logger.dump_tabular()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DQN on cartpole using a custom mlp")
    parser.add_argument('--no-render', default=False, action="store_true", help="Disable rendering")
    parser.add_argument('--max-timesteps', default=50000, type=int,
                        help="Maximum number of timesteps when not rendering")
    args = parser.parse_args()
    main(args)
