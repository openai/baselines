#!/usr/bin/env python
from mpi4py import MPI
from baselines.common import set_global_seeds
from baselines import bench
from baselines.common.mpi_fork import mpi_fork
import os.path as osp
import gym, logging
from baselines import logger
import sys

def wrap_train(env):
    from baselines.common.atari_wrappers import (wrap_deepmind, FrameStack)
    env = wrap_deepmind(env, clip_rewards=True)
    env = FrameStack(env, 4)
    return env

def train(env_id, num_timesteps, seed, num_cpu):
    from baselines.pposgd import pposgd_simple, cnn_policy
    import baselines.common.tf_util as U
    whoami  = mpi_fork(num_cpu)
    if whoami == "parent": return
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    if rank != 0: logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env = gym.make(env_id)
    def policy_fn(name, ob_space, ac_space): #pylint: disable=W0613
        return cnn_policy.CnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space)
    env = bench.Monitor(env, osp.join(logger.get_dir(), "%i.monitor.json" % rank))
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    env = wrap_train(env)
    num_timesteps /= 4 # because we're wrapping the envs to do frame skip
    env.seed(workerseed)

    pposgd_simple.learn(env, policy_fn,
        max_timesteps=num_timesteps,
        timesteps_per_batch=256,
        clip_param=0.2, entcoeff=0.01,
        optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=64,
        gamma=0.99, lam=0.95,
        schedule='linear'
    )
    env.close()

def main():
    train('PongNoFrameskip-v4', num_timesteps=40e6, seed=0, num_cpu=8)

if __name__ == '__main__':
    main()
