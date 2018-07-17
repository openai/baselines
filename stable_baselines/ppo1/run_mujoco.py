#!/usr/bin/env python3

from stable_baselines.ppo1 import mlp_policy, pposgd_simple
from stable_baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from stable_baselines import logger


def train(env_id, num_timesteps, seed):
    """
    Train PPO1 model for the Mujoco environment, for testing purposes

    :param env_id: (str) Environment ID
    :param num_timesteps: (int) The total number of samples
    :param seed: (int) The initial seed for training
    """
    def policy_fn(name, ob_space, ac_space, sess=None, placeholders=None):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, hid_size=64, num_hid_layers=2,
                                    sess=sess, placeholders=placeholders)

    env = make_mujoco_env(env_id, seed)
    pposgd_simple.learn(env, policy_fn,
                        max_timesteps=num_timesteps,
                        timesteps_per_actorbatch=2048,
                        clip_param=0.2, entcoeff=0.0,
                        optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
                        gamma=0.99, lam=0.95, schedule='linear')
    env.close()


def main():
    """
    Runs the test
    """
    args = mujoco_arg_parser().parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == '__main__':
    main()
