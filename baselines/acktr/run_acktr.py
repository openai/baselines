#!/usr/bin/env python
import argparse
import logging
import os
import tensorflow as tf
import gym
from gym import utils

from baselines import logger
from baselines.common import set_global_seeds
from baselines import bench
from baselines.acktr.acktr_cont import learn
from baselines.acktr.policies import GaussianMlpPolicy
from baselines.acktr.value_functions import NeuralNetValueFunction

def train(env_id, num_timesteps, timesteps_per_batch, seed, num_cpu, resume, 
          agentName, logdir, desired_kl, gamma, lam,
          portnum
):
    env=gym.make(env_id)
    if logger.get_dir():
        env = bench.Monitor(env, os.path.join(logger.get_dir(), "monitor.json"))
    set_global_seeds(seed)
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)

    with tf.Session(config=tf.ConfigProto()) as session:
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        with tf.variable_scope("vf"):
            vf = NeuralNetValueFunction(ob_dim, ac_dim)
        with tf.variable_scope("pi"):
            policy = GaussianMlpPolicy(ob_dim, ac_dim)

        learn(env, policy=policy, vf=vf,
            gamma=0.99, lam=0.97, timesteps_per_batch=2500,
            desired_kl=0.002,
            num_timesteps=num_timesteps, animate=False)

        env.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-id', type=str, default='Custom0-v0')
    parser.add_argument('--num-cpu', type=int, default=1)
    parser.add_argument('--seed', type=int, default=57)
    parser.add_argument('--logdir', type=str, default='.')
    parser.add_argument('--agentName',type=str, default='ACKTR-Agent')
    parser.add_argument('--resume',type=int, default = 0)

    parser.add_argument('--num_timesteps', type=int, default = 1e6)
    parser.add_argument('--timesteps_per_batch', type=int, default=2500)
    parser.add_argument('--desired_kl', type=float, default=0.002)

    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)

    parser.add_argument("--portnum", required=False, type=int, default=5000)
    parser.add_argument("--server_ip", required=False, default="localhost")

    return vars(parser.parse_args())

if __name__ == "__main__":

    args = parse_args()
    utils.portnum = args['portnum']
    utils.server_ip = args['server_ip']
    del args['portnum']
    del args['server_ip']

    train(args['env_id'], num_timesteps=args['num_timesteps'], timesteps_per_batch=args['timesteps_per_batch'],
          seed=args['seed'], num_cpu=args['num_cpu'], resume=args['resume'], agentName=args['agentName'], 
          logdir=args['logdir'],desired_kl=args['desired_kl'], gamma=args['gamma'], lam=args['lam'], 
          portnum=utils.portnum,
          )
