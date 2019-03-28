"""
Helpers for scripts like run_atari.py.
"""

import os

from mpi4py import MPI
import gym
from gym.wrappers import FlattenDictWrapper

from stable_baselines import logger
from stable_baselines.bench import Monitor
from stable_baselines.common import set_global_seeds
from stable_baselines.common.atari_wrappers import make_atari, wrap_deepmind
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv


def make_atari_env(env_id, num_env, seed, wrapper_kwargs=None,
                   start_index=0, allow_early_resets=True, start_method=None):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param wrapper_kwargs: (dict) the parameters for wrap_deepmind function
    :param start_index: (int) start rank index
    :param allow_early_resets: (bool) allows early reset of the environment
    :return: (Gym Environment) The atari environment
    :param start_method: (str) method used to start the subprocesses.
        See SubprocVecEnv doc for more information
    """
    if wrapper_kwargs is None:
        wrapper_kwargs = {}

    def make_env(rank):
        def _thunk():
            env = make_atari(env_id)
            env.seed(seed + rank)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
                          allow_early_resets=allow_early_resets)
            return wrap_deepmind(env, **wrapper_kwargs)
        return _thunk
    set_global_seeds(seed)

    # When using one environment, no need to start subprocesses
    if num_env == 1:
        return DummyVecEnv([make_env(0)])

    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)],
                         start_method=start_method)


def make_mujoco_env(env_id, seed, allow_early_resets=True):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param allow_early_resets: (bool) allows early reset of the environment
    :return: (Gym Environment) The mujoco environment
    """
    rank = MPI.COMM_WORLD.Get_rank()
    set_global_seeds(seed + 10000 * rank)
    env = gym.make(env_id)
    env = Monitor(env, os.path.join(logger.get_dir(), str(rank)), allow_early_resets=allow_early_resets)
    env.seed(seed)
    return env


def make_robotics_env(env_id, seed, rank=0, allow_early_resets=True):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) the rank of the environment (for logging)
    :param allow_early_resets: (bool) allows early reset of the environment
    :return: (Gym Environment) The robotic environment
    """
    set_global_seeds(seed)
    env = gym.make(env_id)
    env = FlattenDictWrapper(env, ['observation', 'desired_goal'])
    env = Monitor(
        env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
        info_keywords=('is_success',), allow_early_resets=allow_early_resets)
    env.seed(seed)
    return env


def arg_parser():
    """
    Create an empty argparse.ArgumentParser.

    :return: (ArgumentParser)
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def atari_arg_parser():
    """
    Create an argparse.ArgumentParser for run_atari.py.

    :return: (ArgumentParser) parser {'--env': 'BreakoutNoFrameskip-v4', '--seed': 0, '--num-timesteps': int(1e7)}
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e7))
    return parser


def mujoco_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.

    :return:  (ArgumentParser) parser {'--env': 'Reacher-v2', '--seed': 0, '--num-timesteps': int(1e6), '--play': False}
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='Reacher-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    parser.add_argument('--play', default=False, action='store_true')
    return parser


def robotics_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.

    :return: (ArgumentParser) parser {'--env': 'FetchReach-v0', '--seed': 0, '--num-timesteps': int(1e6)}
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='FetchReach-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    return parser
