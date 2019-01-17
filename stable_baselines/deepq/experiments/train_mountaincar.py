import argparse

import gym

from stable_baselines.deepq import DQN


def main(args):
    """
    Train and save the DQN model, for the mountain car problem

    :param args: (ArgumentParser) the input arguments
    """
    env = gym.make("MountainCar-v0")

    # using layer norm policy here is important for parameter space noise!
    model = DQN(
        policy="LnMlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.1,
        param_noise=True,
        policy_kwargs=dict(layers=[64])
    )
    model.learn(total_timesteps=args.max_timesteps)

    print("Saving model to mountaincar_model.pkl")
    model.save("mountaincar_model.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DQN on cartpole")
    parser.add_argument('--max-timesteps', default=100000, type=int, help="Maximum number of timesteps")
    args = parser.parse_args()
    main(args)
