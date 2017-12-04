#!/usr/bin/env python3
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import gym, logging
from baselines import logger

def train(env_id, num_timesteps, seed):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)
    env = gym.make(env_id)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    env = bench.Monitor(env, logger.get_dir())
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear',
        )
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Hopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    args = parser.parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == '__main__':
    main()
