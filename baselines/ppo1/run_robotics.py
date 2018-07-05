#!/usr/bin/env python3

from mpi4py import MPI
import mujoco_py

from baselines.common import set_global_seeds
from baselines.common.cmd_util import make_robotics_env, robotics_arg_parser
from baselines.ppo1 import mlp_policy, pposgd_simple
import baselines.common.tf_util as tf_util


def train(env_id, num_timesteps, seed):

    rank = MPI.COMM_WORLD.Get_rank()
    with tf_util.single_threaded_session():
        with mujoco_py.ignore_mujoco_warnings():
            workerseed = seed + 10000 * rank
            set_global_seeds(workerseed)
            env = make_robotics_env(env_id, workerseed, rank=rank)

            def policy_fn(name, ob_space, ac_space):
                return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, hid_size=256, num_hid_layers=3)

            pposgd_simple.learn(env, policy_fn,
                                max_timesteps=num_timesteps,
                                timesteps_per_actorbatch=2048,
                                clip_param=0.2, entcoeff=0.0,
                                optim_epochs=5, optim_stepsize=3e-4, optim_batchsize=256,
                                gamma=0.99, lam=0.95, schedule='linear')
            env.close()


def main():
    args = robotics_arg_parser().parse_args()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == '__main__':
    main()
