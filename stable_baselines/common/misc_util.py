import random

import gym
import numpy as np
import tensorflow as tf


def zipsame(*seqs):
    """
    Performes a zip function, but asserts that all zipped elements are of the same size

    :param seqs: a list of arrays that are zipped together
    :return: the zipped arguments
    """
    length = len(seqs[0])
    assert all(len(seq) == length for seq in seqs[1:])
    return zip(*seqs)


def set_global_seeds(seed):
    """
    set the seed for python random, tensorflow, numpy and gym spaces

    :param seed: (int) the seed
    """
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # prng was removed in latest gym version
    if hasattr(gym.spaces, 'prng'):
        gym.spaces.prng.seed(seed)


def boolean_flag(parser, name, default=False, help_msg=None):
    """
    Add a boolean flag to argparse parser.

    :param parser: (argparse.Parser) parser to add the flag to
    :param name: (str) --<name> will enable the flag, while --no-<name> will disable it
    :param default: (bool) default value of the flag
    :param help_msg: (str) help string for the flag
    """
    dest = name.replace('-', '_')
    parser.add_argument("--" + name, action="store_true", default=default, dest=dest, help=help_msg)
    parser.add_argument("--no-" + name, action="store_false", dest=dest)


def mpi_rank_or_zero():
    """
    Return the MPI rank if mpi is installed. Otherwise, return 0.
    :return: (int)
    """
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_rank()
    except ImportError:
        return 0
