import argparse

import gym

from baselines import deepq


def main(args):
    """
    train and save the DeepQ model, for the mountain car problem

    :param args: (ArgumentParser) the input arguments
    """
    env = gym.make("MountainCar-v0")
    # Enabling layer_norm here is import for parameter space noise!
    model = deepq.models.mlp([64], layer_norm=True)
    act = deepq.learn(
        env,
        q_func=model,
        learning_rate=1e-3,
        max_timesteps=args.max_timesteps,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.1,
        print_freq=10,
        param_noise=True
    )
    print("Saving model to mountaincar_model.pkl")
    act.save("mountaincar_model.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DQN on cartpole")
    parser.add_argument('--max-timesteps', default=100000, type=int, help="Maximum number of timesteps")
    args = parser.parse_args()
    main(args)
