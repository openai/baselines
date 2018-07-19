import argparse

import gym

from baselines.deepq import DeepQ, models as deepq_models


def main(args):
    """
    train and save the DeepQ model, for the mountain car problem

    :param args: (ArgumentParser) the input arguments
    """
    env = gym.make("MountainCar-v0")
    # Enabling layer_norm here is import for parameter space noise!
    q_func = deepq_models.mlp([64], layer_norm=True)

    model = DeepQ(
        q_func=q_func,
        env=env,
        learning_rate=1e-3,
        max_timesteps=args.max_timesteps,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.1,
        param_noise=True
    )
    model.learn()

    print("Saving model to mountaincar_model.pkl")
    model.save("mountaincar_model.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DQN on cartpole")
    parser.add_argument('--max-timesteps', default=100000, type=int, help="Maximum number of timesteps")
    args = parser.parse_args()
    main(args)
