#!/usr/bin/env python3
from baselines import logger
from baselines.acer.acer_simple import learn
from baselines.acer.policies import AcerCnnPolicy, AcerLstmPolicy
from baselines.common.cmd_util import make_atari_env, atari_arg_parser


def train(env_id, num_timesteps, seed, policy, lrschedule, num_cpu):
    """
    train an ACER model on atari
    :param env_id: (str) Environment ID
    :param num_timesteps: (int) The total number of samples
    :param seed: (int) The initial seed for training
    :param policy: (A2CPolicy) The policy model to use (MLP, CNN, LSTM, ...)
    :param lrschedule: (str) The type of scheduler for the learning rate update ('linear', 'constant',
                                 'double_linear_con', 'middle_drop' or 'double_middle_drop')
    :param num_cpu: (int) The number of cpu to train on
    """
    env = make_atari_env(env_id, num_cpu, seed)
    if policy == 'cnn':
        policy_fn = AcerCnnPolicy
    elif policy == 'lstm':
        policy_fn = AcerLstmPolicy
    else:
        print("Policy {} not implemented".format(policy))
        return
    learn(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.1), lrschedule=lrschedule, buffer_size=5000)
    env.close()


def main():
    """
    Runs the test
    """
    parser = atari_arg_parser()
    parser.add_argument('--policy', choices=['cnn', 'lstm', 'lnlstm'], default='cnn', help='Policy architecture')
    parser.add_argument('--lrschedule', choices=['constant', 'linear'], default='constant',
                        help='Learning rate schedule')
    parser.add_argument('--logdir', help='Directory for logging')
    args = parser.parse_args()
    logger.configure(args.logdir)
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
          policy=args.policy, lrschedule=args.lrschedule, num_cpu=16)


if __name__ == '__main__':
    main()
