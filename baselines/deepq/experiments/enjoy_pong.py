import gym

from baselines import deepq
from baselines.deepq import DeepQ


def main():
    """
    run a trained model for the pong problem
    """
    env = gym.make("PongNoFrameskip-v4")
    env = deepq.wrap_atari_dqn(env)
    model = DeepQ.load("pong_model.pkl", env)

    with model.sess.as_default():
        while True:
            obs, done = env.reset(), False
            episode_rew = 0
            while not done:
                env.render()
                obs, rew, done, _ = env.step(model.act(obs[None])[0])
                episode_rew += rew
            print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
