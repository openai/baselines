#!/usr/bin/env python3

"""
See trained A2C model in action.
The parameters (env id, policy architecture, etc.) should be the same as in train_atari.py

"""


import time

import tensorflow as tf

from baselines import logger
from baselines.a2c.a2c import Model
from baselines.common.cmd_util import atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy


def enjoy(env_id, seed, policy, model_filename, fps=100):
    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'lnlstm':
        policy_fn = LnLstmPolicy

    env = wrap_deepmind(make_atari(env_id), clip_rewards=False, frame_stack=True)
    env.seed(seed)

    tf.reset_default_graph()
    ob_space = env.observation_space
    ac_space = env.action_space
    nsteps = 5  # default value, change if needed

    model = Model(policy=policy_fn, ob_space=ob_space, ac_space=ac_space, nenvs=1, nsteps=nsteps)
    model.load(model_filename)

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            time.sleep(1.0 / fps)
            action, _, _, _ = model.step_model.step([obs.__array__()])
            obs, rew, done, _ = env.step(action)
            episode_rew += rew
        print('Episode reward:', episode_rew)

    env.close()


def main():
    parser = atari_arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--model-filename', help='Trained model filename', default='atari_a2c.gz')
    args = parser.parse_args()
    logger.configure()
    enjoy(args.env, args.seed, args.policy, args.model_filename)


if __name__ == '__main__':
    main()
