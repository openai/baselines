# !/usr/bin/env python
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import gym, logging
from baselines import logger
import sys
import tensorflow as tf
import roboschool
from argparse import ArgumentParser

def enjoy(env_id, num_timesteps, seed):
    from baselines.pposgd import mlp_policy, pposgd_simple
    sess = U.make_session(num_cpu=1)
    sess.__enter__()
    logger.session().__enter__()
    set_global_seeds(seed)
    env = gym.make(env_id)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    env = bench.Monitor(env, osp.join(logger.get_dir(), "monitor.json"))
    obs = env.reset()
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    pi = policy_fn('pi', env.observation_space, env.action_space)
    tf.train.Saver().restore(sess, '/tmp/model')
    done = False
    while not done:
        action = pi.act(True, obs)[0]
        obs, reward, done, info = env.step(action)
        env.render()

def main():
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1')
    enjoy(parser.parse_args().env, num_timesteps=1e6, seed=0)


if __name__ == '__main__':
    main()
