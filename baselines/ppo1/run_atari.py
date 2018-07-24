#!/usr/bin/env python3
import os

from mpi4py import MPI

from baselines.common import set_global_seeds
from baselines import bench, logger
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.cmd_util import atari_arg_parser
from baselines.ppo1 import PPO1, cnn_policy


def train(env_id, num_timesteps, seed):
    """
    Train PPO1 model for Atari environments, for testing purposes

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

    def policy_fn(name, ob_space, ac_space, sess=None, placeholders=None):  # pylint: disable=W0613
        return cnn_policy.CnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space, sess=sess,
                                    placeholders=placeholders)

    env = bench.Monitor(env, logger.get_dir() and
                        os.path.join(logger.get_dir(), str(rank)))
    env.seed(workerseed)

    env = wrap_deepmind(env)
    env.seed(workerseed)

    model = PPO1(policy_fn, env, timesteps_per_actorbatch=256, clip_param=0.2, entcoeff=0.01, optim_epochs=4,
                 optim_stepsize=1e-3, optim_batchsize=64, gamma=0.99, lam=0.95, schedule='linear')
    model.learn(total_timesteps=num_timesteps)
    env.close()


def main():
    """
    Runs the test
    """
    args = atari_arg_parser().parse_args()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == '__main__':
    main()
