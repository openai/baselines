#!/usr/bin/env python
import argparse
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
from baselines.common.mpi_fork import mpi_fork

import os.path as osp
import gym, logging
from mpi4py import MPI
from gym import utils

from baselines import logger
import sys

def train(env_id, num_timesteps, seed, num_cpu, resume, agentName, logdir,portnum,
          timesteps_per_batch,hid_size,num_hid_layers,clip_param,entcoeff,optim_epochs,optim_stepsize,optim_batchsize,gamma,lam
):
    from baselines.pposgd import mlp_policy, pposgd_simple
    print("num cpu = " + str(num_cpu))
    whoami  = mpi_fork(num_cpu)
    if whoami == "parent": return
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    logger.session().__enter__()
    if rank != 0: logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    utils.portnum = portnum+rank
    env = gym.make(env_id)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=hid_size, num_hid_layers=num_hid_layers)
    env = bench.Monitor(env, osp.join(logger.get_dir(), "monitor.json"))
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    pposgd_simple.learn(env, policy_fn, 
            max_timesteps=num_timesteps,
            timesteps_per_batch=timesteps_per_batch,
            clip_param=clip_param, entcoeff=entcoeff,
            optim_epochs=optim_epochs, optim_stepsize=optim_stepsize, optim_batchsize=optim_batchsize,
            gamma=gamma, lam=lam,
            resume=resume, agentName=agentName, logdir=logdir
        )
    env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--portnum", required=False, type=int, default=5000)
    parser.add_argument("--server_ip", required=False, default="localhost")
    parser.add_argument('--env-id', type=str, default='MountainCarContinuous-v0') #'Humanoid2-v1') # 'Walker2d2-v1')
    parser.add_argument('--num-cpu', type=int, default=1)
    parser.add_argument('--seed', type=int, default=57)
    parser.add_argument('--logdir', type=str, default='.') #default=None)
    parser.add_argument('--agentName',type=str,default='PPO-Agent')
    parser.add_argument('--resume',type=int,default = 0)

    parser.add_argument('--num_timesteps',type=int,default = 1e6)
    parser.add_argument('--timesteps_per_batch', type=int, default=1000)
    parser.add_argument('--hid_size', type=int, default=64)
    parser.add_argument('--num_hid_layers', type=int, default=2)
    parser.add_argument('--clip_param', type=float, default=0.2)
    parser.add_argument('--entcoeff', type=float, default=0.0)
    parser.add_argument('--optim_epochs', type=int, default=20)
    parser.add_argument('--optim_stepsize', type=float, default=3e-4)
    parser.add_argument('--optim_batchsize', type=int, default=64)

    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)
    return vars(parser.parse_args())

def main():
    args = parse_args()
    utils.portnum = args['portnum']
    utils.server_ip = args['server_ip']
    del args['portnum']
    del args['server_ip']

    train(args['env_id'], num_timesteps=args['num_timesteps'], seed=args['seed'], num_cpu=args['num_cpu'], resume=args['resume'], agentName=args['agentName'], logdir=args['logdir'], portnum=utils.portnum,
          timesteps_per_batch=args['timesteps_per_batch'],
          hid_size=args['hid_size'],
          num_hid_layers=args['num_hid_layers'],
          clip_param=args['clip_param'], entcoeff=args['entcoeff'],
          optim_epochs=args['optim_epochs'], optim_stepsize=args['optim_stepsize'], optim_batchsize=args['optim_batchsize'],
          gamma=args['gamma'], lam=args['lam'],
          )


if __name__ == '__main__':
    main()
