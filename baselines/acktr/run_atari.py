#!/usr/bin/env python
import os, logging, gym
from baselines import logger
from baselines.common import set_global_seeds
from baselines import bench
from baselines.acktr.acktr_disc import learn
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import wrap_deepmind
from baselines.acktr.policies import CnnPolicy

def train(env_id, num_timesteps, seed, num_cpu):
    num_timesteps //= 4

    def make_env(rank):
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            if logger.get_dir():
                env = bench.Monitor(env, os.path.join(logger.get_dir(), "{}.monitor.json".format(rank)))
            gym.logger.setLevel(logging.WARN)
            return wrap_deepmind(env)
        return _thunk

    set_global_seeds(seed)
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    policy_fn = CnnPolicy
    learn(policy_fn, env, seed, total_timesteps=num_timesteps, nprocs=num_cpu)
    env.close()

def main():
    train('BreakoutNoFrameskip-v4', num_timesteps=int(40e6), seed=0, num_cpu=32)


if __name__ == '__main__':
    main()
