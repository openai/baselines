#!/usr/bin/env python3

import tensorflow as tf

from stable_baselines import logger
from stable_baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from stable_baselines.acktr.acktr_cont import learn
from stable_baselines.acktr.policies import GaussianMlpPolicy
from stable_baselines.acktr.value_functions import NeuralNetValueFunction


def train(env_id, num_timesteps, seed):
    """
    train an ACKTR model on atari

    :param env_id: (str) Environment ID
    :param num_timesteps: (int) The total number of samples
    :param seed: (int) The initial seed for training
    """
    env = make_mujoco_env(env_id, seed)

    with tf.Session(config=tf.ConfigProto()):
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        with tf.variable_scope("vf"):
            value_fn = NeuralNetValueFunction(ob_dim, ac_dim)
        with tf.variable_scope("pi"):
            policy = GaussianMlpPolicy(ob_dim, ac_dim)

        learn(env, policy=policy, value_fn=value_fn, gamma=0.99, lam=0.97, timesteps_per_batch=2500, desired_kl=0.002,
              num_timesteps=num_timesteps, animate=False)

        env.close()


def main():
    """
    Runs the test
    """
    args = mujoco_arg_parser().parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == "__main__":
    main()
