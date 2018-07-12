#!/usr/bin/env python3

from baselines import logger
from baselines.acktr.acktr_disc import learn
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.a2c.policies import CnnPolicy


def train(env_id, num_timesteps, seed, num_cpu):
    """
    train an ACKTR model on atari

    :param env_id: (str) Environment ID
    :param num_timesteps: (int) The total number of samples
    :param seed: (int) The initial seed for training
    :param num_cpu: (int) The number of cpu to train on
    """
    env = VecFrameStack(make_atari_env(env_id, num_cpu, seed), 4)
    learn(CnnPolicy, env, seed, total_timesteps=int(num_timesteps * 1.1), nprocs=num_cpu)
    env.close()


def main():
    """
    Runs the test
    """
    args = atari_arg_parser().parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, num_cpu=32)


if __name__ == '__main__':
    main()
