import argparse

import gym
import numpy as np

from stable_baselines.deepq import DQN


def main(args):
    """
    Run a trained model for the mountain car problem

    :param args: (ArgumentParser) the input arguments
    """
    env = gym.make("MountainCar-v0")
    model = DQN.load("mountaincar_model.pkl", env)

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            if not args.no_render:
                env.render()
            # Epsilon-greedy
            if np.random.random() < 0.02:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, rew, done, _ = env.step(action)
            episode_rew += rew
        print("Episode reward", episode_rew)
        # No render is only used for automatic testing
        if args.no_render:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Enjoy trained DQN on MountainCar")
    parser.add_argument('--no-render', default=False, action="store_true", help="Disable rendering")
    args = parser.parse_args()
    main(args)
