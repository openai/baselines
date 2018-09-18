import argparse

from stable_baselines import bench, logger
from stable_baselines.common import set_global_seeds
from stable_baselines.common.atari_wrappers import make_atari
from stable_baselines.deepq import DQN, wrap_atari_dqn, CnnPolicy


def main():
    """
    Run the atari test
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--prioritized', type=int, default=1)
    parser.add_argument('--prioritized-replay-alpha', type=float, default=0.6)
    parser.add_argument('--dueling', type=int, default=1)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    parser.add_argument('--checkpoint-freq', type=int, default=10000)
    parser.add_argument('--checkpoint-path', type=str, default=None)

    args = parser.parse_args()
    logger.configure()
    set_global_seeds(args.seed)
    env = make_atari(args.env)
    env = bench.Monitor(env, logger.get_dir())
    env = wrap_atari_dqn(env)

    model = DQN(
        env=env,
        policy=CnnPolicy,
        learning_rate=1e-4,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=bool(args.prioritized),
        prioritized_replay_alpha=args.prioritized_replay_alpha,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_path=args.checkpoint_path,
    )
    model.learn(total_timesteps=args.num_timesteps)

    env.close()


if __name__ == '__main__':
    main()
