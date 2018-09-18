import gym

from stable_baselines import deepq
from stable_baselines.deepq import DQN


def main():
    """
    Run a trained model for the pong problem
    """
    env = gym.make("PongNoFrameskip-v4")
    env = deepq.wrap_atari_dqn(env)
    model = DQN.load("pong_model.pkl", env)

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            action, _ = model.predict(obs)
            obs, rew, done, _ = env.step(action)
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
