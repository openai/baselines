""" Use a pre-trained acktr model to play Breakout.
    To train: python3 ./run_atari.py
        You'll need to add "logger.configure(<some dir>)" to run_atari.py so it will save checkpoint files
    Then run this script with a checkpoint file as the argument
    A running average of the past 100 rewards will be printed
"""
from optparse import OptionParser
from collections import deque

import gym
import cloudpickle
import numpy as np
import tensorflow as tf

from baselines.common.atari_wrappers import wrap_deepmind
from baselines.acktr.acktr_disc import Model
from baselines.acktr.policies import CnnPolicy
from baselines.common import set_global_seeds, explained_variance

def getOptions():
    usage = "Usage: python3 enjoy_breakout.py [options] <checkpoint>"
    parser = OptionParser( usage=usage )
    parser.add_option("-r","--render", action="store_true", default=False, help="Render gym environment. Will greatly reduce speed.");
    parser.add_option("-m","--max_episodes", default="0", type="int", help="Maximum number of episodes to play.");

    (options, args) = parser.parse_args()

    if len(args) != 1:
        print( usage )
        exit()

    return (options, args)

def update_obs(state, obs):
    obs = np.reshape( obs, state.shape[0:3] )
    state = np.roll(state, shift=-1, axis=3)
    state[:, :, :, -1] = obs
    return state

def main():
    options, args = getOptions()

    env = gym.make("BreakoutNoFrameskip-v4")
    env = wrap_deepmind(env)

    tf.reset_default_graph()
    set_global_seeds(0)

    total_timesteps=int(40e6)
    nprocs = 2
    nenvs = 1
    nstack = 4
    nsteps = 1
    nenvs = 1

    ob_space = env.observation_space
    ac_space = env.action_space
    nh, nw, nc = ob_space.shape
    batch_ob_shape = (nenvs*nsteps, nh, nw, nc*nstack)

    policy=CnnPolicy
    model = Model(policy, ob_space, ac_space, nenvs, total_timesteps, nprocs=nprocs, nstack=nstack)
    model.load(args[0])
    act = model.step_model
    
    episode = 1
    reward_100 = deque(maxlen=100)

    while options.max_episodes == 0 or episode <= options.max_episodes:
        state = np.zeros(batch_ob_shape, dtype=np.uint8)
        states = model.initial_state

        obs, done = env.reset(), False
        episode_reward = 0
        while not done:
            if options.render:
                env.render()
            state = update_obs(state,obs)

            actions, values, states = act.step(state, states, [done])
            obs, rew, done, _ = env.step(actions[0])

            episode_reward += rew
        reward_100.append(episode_reward)
        print( "Reward/Avg100: {0}  {1:.3f}".format( episode_reward, np.mean(reward_100) ) )
        episode += 1


if __name__ == '__main__':
    main()
