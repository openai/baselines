#!/usr/bin/env python
from mpi4py import MPI
from baselines.common import set_global_seeds
import os.path as osp
import gym, logging
from baselines import logger
from baselines import bench
import sys

def wrap_train(env):
    from baselines.common.atari_wrappers import (wrap_deepmind, FrameStack)
    env = wrap_deepmind(env, clip_rewards=False)
    env = FrameStack(env, 3)
    return env

def train(env_id, num_frames, seed):
    from baselines.trpo_mpi.nosharing_cnn_policy import CnnPolicy
    from baselines.trpo_mpi import trpo_mpi
    import baselines.common.tf_util as U
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    if rank != 0:
        logger.set_level(logger.DISABLED)


    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env = gym.make(env_id)
    def policy_fn(name, ob_space, ac_space): #pylint: disable=W0613
        return CnnPolicy(name=name, ob_space=env.observation_space, ac_space=env.action_space)
    env = bench.Monitor(env, logger.get_dir() and 
        osp.join(logger.get_dir(), "%i.monitor.json"%rank))
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    env = wrap_train(env)
    num_timesteps = int(num_frames / 4 * 1.1)
    env.seed(workerseed)

    trpo_mpi.learn(env, policy_fn, timesteps_per_batch=512, max_kl=0.001, cg_iters=10, cg_damping=1e-3,
        max_timesteps=num_timesteps, gamma=0.98, lam=1.0, vf_iters=3, vf_stepsize=1e-4, entcoeff=0.00)
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='PongNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    args = parser.parse_args()
    train(args.env, num_frames=40e6, seed=args.seed)


if __name__ == "__main__":
    main()
