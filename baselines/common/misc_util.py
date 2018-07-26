import os
import pickle
import random
import tempfile
import zipfile

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


def unpack(seq, sizes):
    """
    Unpack 'seq' into a sequence of lists, with lengths specified by 'sizes'.
    None = just one bare element, not a list

    Example:
    unpack([1,2,3,4,5,6], [3,None,2]) -> ([1,2,3], 4, [5,6])

    :param seq: (Iterable) the sequence to unpack
    :param sizes: ([int]) the shape to unpack
    :return: ([Any] or Any) the unpacked sequence
    """
    seq = list(seq)
    iterator = iter(seq)
    assert sum(1 if s is None else s for s in sizes) == len(seq), "Trying to unpack %s into %s" % (seq, sizes)
    for size in sizes:
        if size is None:
            yield iterator.__next__()
        else:
            _list = []
            for _ in range(size):
                _list.append(iterator.__next__())
            yield _list


class EzPickle(object):
    def __init__(self, *args, **kwargs):
        """
        Objects that are pickled and unpickled via their constructor arguments.

        Example usage:

            class Dog(Animal, EzPickle):
                def __init__(self, furcolor, tailkind="bushy"):
                    Animal.__init__()
                    EzPickle.__init__(furcolor, tailkind)
                    ...

        When this object is unpickled, a new Dog will be constructed by passing the provided
        furcolor and tailkind into the constructor. However, philosophers are still not sure
        whether it is still the same dog.

        This is generally needed only for environments which wrap C/C++ code, such as MuJoCo
        and Atari.

        :param args: ezpickle args
        :param kwargs: ezpickle kwargs
        """
        self._ezpickle_args = args
        self._ezpickle_kwargs = kwargs

    def __getstate__(self):
        return {"_ezpickle_args": self._ezpickle_args, "_ezpickle_kwargs": self._ezpickle_kwargs}

    def __setstate__(self, _dict):
        out = type(self)(*_dict["_ezpickle_args"], **_dict["_ezpickle_kwargs"])
        self.__dict__.update(out.__dict__)


def set_global_seeds(seed):
    """
    set the seed for python random, tensorflow, numpy and gym spaces

    :param seed: (int) the seed
    """
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    gym.spaces.prng.seed(seed)


def pretty_eta(seconds_left):
    """
    Print the number of seconds in human readable format.

    Examples:
    2 days
    2 hours and 37 minutes
    less than a minute

    :param seconds_left: (int) Number of seconds to be converted to the ETA
    :return: (str) String representing the pretty ETA.
    """
    minutes_left = seconds_left // 60
    seconds_left %= 60
    hours_left = minutes_left // 60
    minutes_left %= 60
    days_left = hours_left // 24
    hours_left %= 24

    def helper(cnt, name):
        return "{} {}{}".format(str(cnt), name, ('s' if cnt > 1 else ''))

    if days_left > 0:
        msg = helper(days_left, 'day')
        if hours_left > 0:
            msg += ' and ' + helper(hours_left, 'hour')
        return msg
    if hours_left > 0:
        msg = helper(hours_left, 'hour')
        if minutes_left > 0:
            msg += ' and ' + helper(minutes_left, 'minute')
        return msg
    if minutes_left > 0:
        return helper(minutes_left, 'minute')
    return 'less than a minute'


class RunningAvg(object):
    def __init__(self, gamma, init_value=None):
        """
        Keep a running estimate of a quantity. This is a bit like mean
        but more sensitive to recent changes.

        :param gamma: (float) Must be between 0 and 1, where 0 is the most sensitive to recent changes.
        :param init_value: (float) Initial value of the estimate. If None, it will be set on the first update.
        """
        self._value = init_value
        self._gamma = gamma

    def update(self, new_val):
        """
        Update the estimate.

        :param new_val: (float) new observated value of estimated quantity.
        """
        if self._value is None:
            self._value = new_val
        else:
            self._value = self._gamma * self._value + (1.0 - self._gamma) * new_val

    def __float__(self):
        """
        Get the current estimate

        :return: (float) current value
        """
        return self._value


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


def get_wrapper_by_name(env, classname):
    """
    Given an a gym environment possibly wrapped multiple times, returns a wrapper
    of class named classname or raises ValueError if no such wrapper was applied

    :param env: (Gym Environment) the environment
    :param classname: (str) name of the wrapper
    :return: (Gym Environment) the wrapped environment
    """
    currentenv = env
    while True:
        if classname == currentenv.class_name():
            return currentenv
        elif isinstance(currentenv, gym.Wrapper):
            currentenv = currentenv.env
        else:
            raise ValueError("Couldn't find wrapper named %s" % classname)


def relatively_safe_pickle_dump(obj, path, compression=False):
    """
    This is just like regular pickle dump, except from the fact that failure cases are
    different:

        - It's never possible that we end up with a pickle in corrupted state.
        - If a there was a different file at the path, that file will remain unchanged in the
          even of failure (provided that filesystem rename is atomic).
        - it is sometimes possible that we end up with useless temp file which needs to be
          deleted manually (it will be removed automatically on the next function call)

    The indended use case is periodic checkpoints of experiment state, such that we never
    corrupt previous checkpoints if the current one fails.

    :param obj: (Object) object to pickle
    :param path: (str) path to the output file
    :param compression: (bool) if true pickle will be compressed
    """
    temp_storage = path + ".relatively_safe"
    if compression:
        # Using gzip here would be simpler, but the size is limited to 2GB
        with tempfile.NamedTemporaryFile() as uncompressed_file:
            pickle.dump(obj, uncompressed_file)
            uncompressed_file.file.flush()
            with zipfile.ZipFile(temp_storage, "w", compression=zipfile.ZIP_DEFLATED) as myzip:
                myzip.write(uncompressed_file.name, "data")
    else:
        with open(temp_storage, "wb") as file_handler:
            pickle.dump(obj, file_handler)
    os.rename(temp_storage, path)


def pickle_load(path, compression=False):
    """
    Unpickle a possible compressed pickle.

    :param path: (str) path to the output file
    :param compression: (bool) if true assumes that pickle was compressed when created and attempts decompression.
    :return: (Object) the unpickled object
    """

    if compression:
        with zipfile.ZipFile(path, "r", compression=zipfile.ZIP_DEFLATED) as myzip:
            with myzip.open("data") as file_handler:
                return pickle.load(file_handler)
    else:
        with open(path, "rb") as file_handler:
            return pickle.load(file_handler)
