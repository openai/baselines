from baselines import deepq
from baselines import bench
from baselines import logger
from baselines.common.atari_wrappers import make_atari


def main():
    exp_dir = './runs/pong'

    # by default CSV logs will be created in OS temp directory
    logger.configure(dir=exp_dir, 
        format_strs=['stdout','log','csv','tensorboard'], log_suffix=None)

    # create Atari environment, use no-op reset, max pool last two frames
    env = make_atari('PongNoFrameskip-v4')

    # by default monitor will log episod reward and log
    env = bench.Monitor(env, logger.get_dir())
    env = deepq.wrap_atari_dqn(env)

    model = deepq.learn(
        env,
        "conv_only",
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=True,
        lr=1e-4,
        total_timesteps=int(1e7),
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        checkpoint_path=exp_dir,
        checkpoint_freq=100000
    )

    model.save('pong_model.pkl')
    env.close()

if __name__ == '__main__':
    main()
