import argparse

import gym

from baselines.deepq import DeepQ


def main(args):
    """
    run a trained model for the mountain car problem

    :param args: (ArgumentParser) the input arguments
    """
    env = gym.make("MountainCar-v0")
    model = DeepQ.load("mountaincar_model.pkl", env)

    with model.sess.as_default():
        while True:
            obs, done = env.reset(), False
            episode_rew = 0
            while not done:
                if not args.no_render:
                    env.render()
                obs, rew, done, _ = env.step(model.act(obs[None])[0])
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
