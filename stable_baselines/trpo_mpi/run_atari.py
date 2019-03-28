#!/usr/bin/env python3
import os

from mpi4py import MPI

from stable_baselines.common import set_global_seeds
from stable_baselines import bench, logger, TRPO
from stable_baselines.common.atari_wrappers import make_atari, wrap_deepmind
from stable_baselines.common.cmd_util import atari_arg_parser
from stable_baselines.common.policies import CnnPolicy


def train(env_id, num_timesteps, seed):
    """
    Train TRPO model for the atari environment, for testing purposes

    :param env_id: (str) Environment ID
    :param num_timesteps: (int) The total number of samples
    :param seed: (int) The initial seed for training
    """
    rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])

    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env = make_atari(env_id)

    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
    env.seed(workerseed)

    env = wrap_deepmind(env)
    env.seed(workerseed)

    model = TRPO(CnnPolicy, env, timesteps_per_batch=512, max_kl=0.001, cg_iters=10, cg_damping=1e-3, entcoeff=0.0,
                 gamma=0.98, lam=1, vf_iters=3, vf_stepsize=1e-4)
    model.learn(total_timesteps=int(num_timesteps * 1.1))
    env.close()
    # Free memory
    del env


def main():
    """
    Runs the test
    """
    args = atari_arg_parser().parse_args()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == "__main__":
    main()
