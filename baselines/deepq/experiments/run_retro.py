import argparse

from baselines import deepq
from baselines.common import set_global_seeds
from baselines import bench
from baselines import logger
from baselines.common import retro_wrappers
import retro


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='SuperMarioBros-Nes')
    parser.add_argument('--gamestate', help='game state to load', default='Level1-1')
    parser.add_argument('--seed', help='seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    args = parser.parse_args()
    logger.configure()
    set_global_seeds(args.seed)
    env = retro_wrappers.make_retro(game=args.env, state=args.gamestate, max_episode_steps=10000, use_restricted_actions=retro.Actions.DISCRETE)
    env.seed(args.seed)
    env = bench.Monitor(env, logger.get_dir())
    env = retro_wrappers.wrap_deepmind_retro(env)

    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=True
    )
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-4,
        max_timesteps=args.num_timesteps,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=True
    )
    act.save()
    env.close()


if __name__ == '__main__':
    main()
