#!/usr/bin/env python3

from mpi4py import MPI
import mujoco_py

from baselines.common import set_global_seeds
from baselines.common.cmd_util import make_robotics_env, robotics_arg_parser
from baselines.ppo1 import mlp_policy, PPO1


def train(env_id, num_timesteps, seed):
    """
    Train PPO1 model for Robotics environment, for testing purposes

    :param env_id: (str) Environment ID
    :param num_timesteps: (int) The total number of samples
    :param seed: (int) The initial seed for training
    """

    rank = MPI.COMM_WORLD.Get_rank()
    with mujoco_py.ignore_mujoco_warnings():
        workerseed = seed + 10000 * rank
        set_global_seeds(workerseed)
        env = make_robotics_env(env_id, workerseed, rank=rank)

        def policy_fn(name, ob_space, ac_space, sess=None, placeholders=None):
            return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, hid_size=256, num_hid_layers=3,
                                        sess=sess, placeholders=placeholders)

        model = PPO1(policy_fn, env, timesteps_per_actorbatch=2048, clip_param=0.2, entcoeff=0.0, optim_epochs=5,
                     optim_stepsize=3e-4, optim_batchsize=256, gamma=0.99, lam=0.95, schedule='linear')
        model.learn(total_timesteps=num_timesteps)
        env.close()


def main():
    """
    Runs the test
    """
    args = robotics_arg_parser().parse_args()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == '__main__':
    main()
