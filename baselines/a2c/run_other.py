#!/usr/bin/env python
import os, logging, gym
from baselines import logger
from baselines.common import set_global_seeds
from baselines.common.misc_util import boolean_flag
from baselines import bench
from baselines.bench.episodic_bench import EpisodeCounter
from baselines.a2c.a2c import learn, eval_policy
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.a2c.policies import CnnPolicy, LstmPolicy, LnLstmPolicy
import gym_grasping

def train(env_id, num_timesteps, seed, policy, lrschedule, num_env, eval, save_timesteps, load_snapshot, atari_env, clip_rewards):
    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'lnlstm':
        policy_fn = LnLstmPolicy

    def make_env(rank):
        def _thunk():
            if atari_env:
                env = make_atari(env_id)
            else:
                env = gym.make(env_id)
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            gym.logger.setLevel(logging.WARN)
            # only clip rewards when not evaluating
            return wrap_deepmind(env, episode_life=atari_env, clip_rewards=not eval and clip_rewards)
        return _thunk
    set_global_seeds(seed)
    env = SubprocVecEnv([make_env(i) for i in range(num_env)])
    if eval:
        ec = EpisodeCounter(num_env, bench_dir=logger.get_dir())
        eval_policy(policy_fn, env, seed, ec.step, load_snapshot=load_snapshot)
        snapshot_name = os.path.splitext(os.path.basename(load_snapshot))[0]
        eval_name = os.path.join(logger.get_dir(), snapshot_name + ".csv")
        ec.save_results(eval_name)
    else:
        learn(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.1), lrschedule=lrschedule, load_snapshot=load_snapshot, save_timesteps=save_timesteps)
    env.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Environment
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    boolean_flag(parser, 'atari-env', help='Preprocess frames like deepmind.'
        'Use for atari environments.', default=True)
    boolean_flag(parser, 'clip-rewards', help='Clip rewards during training.', default=True)
    boolean_flag(parser, 'eval', help='Evaluation mode', default=False)
    # Core A2C parameters
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    # Checkpointing
    # save by timesteps, because we evaluate by timesteps
    parser.add_argument('--save-timesteps', help='Model snapshot during training', type=int, default=None)
    parser.add_argument('--load-snapshot', help='Load snapshot', type=str, default='')

    args = parser.parse_args()
    logger.configure()

    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
        policy=args.policy, lrschedule=args.lrschedule, num_env=16,
        eval=args.eval, save_timesteps=args.save_timesteps, load_snapshot=args.load_snapshot,
        atari_env=args.atari_env, clip_rewards=args.clip_rewards)

if __name__ == '__main__':
    main()
