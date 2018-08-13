import argparse

import numpy as np

from baselines import deepq
from baselines.common import retro_wrappers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='environment ID', default='SuperMarioBros-Nes')
    parser.add_argument('--gamestate', help='game state to load', default='Level1-1')
    parser.add_argument('--model', help='model pickle file from ActWrapper.save', default='model.pkl')
    args = parser.parse_args()

    env = retro_wrappers.make_retro(game=args.env, state=args.gamestate, max_episode_steps=None)
    env = retro_wrappers.wrap_deepmind_retro(env)
    act = deepq.load(args.model)

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            action = act(obs[None])[0]
            env_action = np.zeros(env.action_space.n)
            env_action[action] = 1
            obs, rew, done, _ = env.step(env_action)
            episode_rew += rew
        print('Episode reward', episode_rew)


if __name__ == '__main__':
    main()
