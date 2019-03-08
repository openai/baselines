"""
Helpers for dealing with vectorized environments.
"""

from collections import OrderedDict

import gym
import numpy as np


def copy_obs_dict(obs):
    """
    Deep-copy a dict of numpy arrays.

    :param obs: (OrderedDict<ndarray>): a dict of numpy arrays.
    :return (OrderedDict<ndarray>) a dict of copied numpy arrays.
    """
    assert isinstance(obs, OrderedDict), "unexpected type for observations '{}'".format(type(obs))
    return OrderedDict([(k, np.copy(v)) for k, v in obs.items()])


def dict_to_obs(space, obs_dict):
    """
    Convert an internal representation raw_obs into the appropriate type
    specified by space.

    :param space: (gym.spaces.Space) an observation space.
    :param obs_dict: (OrderedDict<ndarray>) a dict of numpy arrays.
    :return (ndarray, tuple<ndarray> or dict<ndarray>): returns an observation
            of the same type as space. If space is Dict, function is identity;
            if space is Tuple, converts dict to Tuple; otherwise, space is
            unstructured and returns the value raw_obs[None].
    """
    if isinstance(space, gym.spaces.Dict):
        return obs_dict
    elif isinstance(space, gym.spaces.Tuple):
        assert len(obs_dict) == len(space.spaces), "size of observation does not match size of observation space"
        return tuple((obs_dict[i] for i in range(len(space.spaces))))
    else:
        assert set(obs_dict.keys()) == {None}, "multiple observation keys for unstructured observation space"
        return obs_dict[None]


def obs_space_info(obs_space):
    """
    Get dict-structured information about a gym.Space.

    Dict spaces are represented directly by their dict of subspaces.
    Tuple spaces are converted into a dict with keys indexing into the tuple.
    Unstructured spaces are represented by {None: obs_space}.

    :param obs_space: (gym.spaces.Space) an observation space
    :return (tuple) A tuple (keys, shapes, dtypes):
        keys: a list of dict keys.
        shapes: a dict mapping keys to shapes.
        dtypes: a dict mapping keys to dtypes.
    """
    if isinstance(obs_space, gym.spaces.Dict):
        assert isinstance(obs_space.spaces, OrderedDict), "Dict space must have ordered subspaces"
        subspaces = obs_space.spaces
    elif isinstance(obs_space, gym.spaces.Tuple):
        subspaces = {i: space for i, space in enumerate(obs_space.spaces)}
    else:
        assert not hasattr(obs_space, 'spaces'), "Unsupported structured space '{}'".format(type(obs_space))
        subspaces = {None: obs_space}
    keys = []
    shapes = {}
    dtypes = {}
    for key, box in subspaces.items():
        keys.append(key)
        shapes[key] = box.shape
        dtypes[key] = box.dtype
    return keys, shapes, dtypes
