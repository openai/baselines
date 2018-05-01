#!/usr/bin/env python3

from functools import partial

from baselines import logger
from baselines.acktr.acktr_disc import learn
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.ppo2.policies import CnnPolicy

def train(env_id, num_timesteps, seed, num_cpu):
    env = VecFrameStack(make_atari_env(env_id, num_cpu, seed), 4)
    policy_fn = partial(CnnPolicy, one_dim_bias=True)
    learn(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.1), nprocs=num_cpu)
    env.close()

def main():
    args = atari_arg_parser().parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, num_cpu=32)

if __name__ == '__main__':
    main()
