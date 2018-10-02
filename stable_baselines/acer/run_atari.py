#!/usr/bin/env python3

from stable_baselines import logger
from stable_baselines.acer import ACER
from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy
from stable_baselines.common.cmd_util import make_atari_env, atari_arg_parser
from stable_baselines.common.vec_env import VecFrameStack


def train(env_id, num_timesteps, seed, policy, lr_schedule, num_cpu):
    """
    train an ACER model on atari

    :param env_id: (str) Environment ID
    :param num_timesteps: (int) The total number of samples
    :param seed: (int) The initial seed for training
    :param policy: (A2CPolicy) The policy model to use (MLP, CNN, LSTM, ...)
    :param lr_schedule: (str) The type of scheduler for the learning rate update ('linear', 'constant',
                                 'double_linear_con', 'middle_drop' or 'double_middle_drop')
    :param num_cpu: (int) The number of cpu to train on
    """
    env = VecFrameStack(make_atari_env(env_id, num_cpu, seed), 4)
    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = CnnLstmPolicy
    else:
        print("Policy {} not implemented".format(policy))
        return

    model = ACER(policy_fn, env, lr_schedule=lr_schedule, buffer_size=5000)
    model.learn(total_timesteps=int(num_timesteps * 1.1), seed=seed)
    env.close()


def main():
    """
    Runs the test
    """
    parser = atari_arg_parser()
    parser.add_argument('--policy', choices=['cnn', 'lstm', 'lnlstm'], default='cnn', help='Policy architecture')
    parser.add_argument('--lr_schedule', choices=['constant', 'linear'], default='constant',
                        help='Learning rate schedule')
    parser.add_argument('--logdir', help='Directory for logging')
    args = parser.parse_args()
    logger.configure(args.logdir)
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
          policy=args.policy, lr_schedule=args.lr_schedule, num_cpu=16)


if __name__ == '__main__':
    main()
