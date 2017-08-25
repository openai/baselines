import opensim as osim
from osim.http.client import Client
from osim.env import *
import numpy as np
import argparse

import tensorflow as tf

from baselines.common import set_global_seeds, tf_util as U
from baselines import bench, logger

from baselines.pposgd import mlp_policy, pposgd_simple

import os.path as osp
import gym, logging
from gym import utils

from baselines import logger
import sys


# Settings
remote_base = 'http://grader.crowdai.org:1729'

# Command line parameters
parser = argparse.ArgumentParser(description='Submit the result to crowdAI')
parser.add_argument('--token', dest='token', action='store', required=True)
parser.add_argument('--logdir', type=str, default='saves')
parser.add_argument('--agentName', type=str, default='PPO-Agent')
parser.add_argument('--hid_size', type=int, default=64)
parser.add_argument('--num_hid_layers', type=int, default=2)
parser.add_argument('--resume', type=int, default=1197)
args = parser.parse_args()

sess = U.single_threaded_session()
sess.__enter__()

def policy_fn(name, ob_space, ac_space):
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
        hid_size=args.hid_size, num_hid_layers=args.num_hid_layers)


env = RunEnv(visualize=False)
ob_space = env.observation_space
ac_space = env.action_space
pi = policy_fn("pi", ob_space, ac_space) # Construct network for the trained policy

ob = U.get_placeholder_cached(name="ob")
ac = pi.pdtype.sample_placeholder([None])

saver = tf.train.Saver()
if args.resume > 0:
    saver.restore(tf.get_default_session(), os.path.join(os.path.abspath(args.logdir), "{}-{}".format(args.agentName, args.resume)))
else:
    print("No weights to load!")

client = Client(remote_base)

# Create environment
observation = client.env_create(args.token)

# Run a single step
#
# The grader runs 3 simulations of at most 1000 steps each. We stop after the last one
while True:
    v = np.array(observation).reshape((env.observation_space.shape[0]))
    action, vpred = pi.act(False, v)
    [observation, reward, done, info] = client.env_step(action.tolist())
    print(observation)
    if done:
        observation = client.env_reset()
        if not observation:
            break

client.submit()
