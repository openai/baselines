#!/usr/bin/env python3
import numpy as np
import gym
import tensorflow as tf

from baselines.common.cmd_util import mujoco_arg_parser
from baselines import bench, logger
from baselines.common import set_global_seeds
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.ppo2 import ppo2
from baselines.a2c.policies import MlpPolicy
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv


def train(env_id, num_timesteps, seed):
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)

    with tf.Session(config=config):
        def make_env():
            env_out = gym.make(env_id)
            env_out = bench.Monitor(env_out, logger.get_dir(), allow_early_resets=True)
            return env_out

        env = DummyVecEnv([make_env])
        env = VecNormalize(env)

        set_global_seeds(seed)
        policy = MlpPolicy
        model = ppo2.learn(policy=policy, env=env, nsteps=2048, nminibatches=32,
                           lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
                           ent_coef=0.0,
                           lr=3e-4,
                           cliprange=0.2,
                           total_timesteps=num_timesteps)

    return model, env


def main():
    args = mujoco_arg_parser().parse_args()
    logger.configure()
    model, env = train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)

    if args.play:
        logger.log("Running trained model")
        obs = np.zeros((env.num_envs,) + env.observation_space.shape)
        obs[:] = env.reset()
        while True:
            actions = model.step(obs)[0]
            obs[:] = env.step(actions)[0]
            env.render()


if __name__ == '__main__':
    main()
