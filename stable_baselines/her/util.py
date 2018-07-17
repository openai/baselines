import os
import subprocess
import sys
import importlib

import tensorflow as tf
import numpy as np
from mpi4py import MPI

from stable_baselines.common import tf_util


def import_function(spec):
    """
    Import a function identified by a string like "pkg.module:fn_name".

    :param spec: (str) the function to import
    :return: (function)
    """
    mod_name, fn_name = spec.split(':')
    module = importlib.import_module(mod_name)
    func = getattr(module, fn_name)
    return func


def flatten_grads(var_list, grads):
    """
    Flattens a variables and their gradients.

    :param var_list: ([TensorFlow Tensor]) the variables
    :param grads: ([TensorFlow Tensor]) the gradients
    :return: (TensorFlow Tensor) the flattend variable and gradient
    """
    return tf.concat([tf.reshape(grad, [tf_util.numel(v)])
                      for (v, grad) in zip(var_list, grads)], 0)


def mlp(_input, layers_sizes, reuse=None, flatten=False, name=""):
    """
    Creates a simple fully-connected neural network

    :param _input: (TensorFlow Tensor) the input
    :param layers_sizes: ([int]) the hidden layers
    :param reuse: (bool) Enable reuse of the network
    :param flatten: (bool) flatten the network output
    :param name: (str) the name of the network
    :return: (TensorFlow Tensor) the network
    """
    for i, size in enumerate(layers_sizes):
        activation = tf.nn.relu if i < len(layers_sizes) - 1 else None
        _input = tf.layers.dense(inputs=_input,
                                 units=size,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 reuse=reuse,
                                 name=name + '_' + str(i))
        if activation:
            _input = activation(_input)
    if flatten:
        assert layers_sizes[-1] == 1
        _input = tf.reshape(_input, [-1])
    return _input


def install_mpi_excepthook():
    """
    setup the MPI exception hooks
    """
    old_hook = sys.excepthook

    def new_hook(a, b, c):
        old_hook(a, b, c)
        sys.stdout.flush()
        sys.stderr.flush()
        MPI.COMM_WORLD.Abort()

    sys.excepthook = new_hook


def mpi_fork(rank, extra_mpi_args=None):
    """
    Re-launches the current script with workers
    Returns "parent" for original parent, "child" for MPI children

    :param rank: (int) the thread rank
    :param extra_mpi_args: (dict) extra arguments for MPI
    :return: (str) the correct type of thread name
    """
    if extra_mpi_args is None:
        extra_mpi_args = []

    if rank <= 1:
        return "child"
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        # "-bind-to core" is crucial for good performance
        args = ["mpirun", "-np", str(rank)] + \
               extra_mpi_args + \
               [sys.executable]

        args += sys.argv
        subprocess.check_call(args, env=env)
        return "parent"
    else:
        install_mpi_excepthook()
        return "child"


def convert_episode_to_batch_major(episode):
    """
    Converts an episode to have the batch dimension in the major (first) dimension.

    :param episode: (dict) the episode batch
    :return: (dict) the episode batch with he batch dimension in the major (first) dimension.
    """
    episode_batch = {}
    for key in episode.keys():
        val = np.array(episode[key]).copy()
        # make inputs batch-major instead of time-major
        episode_batch[key] = val.swapaxes(0, 1)

    return episode_batch


def transitions_in_episode_batch(episode_batch):
    """
    Number of transitions in a given episode batch.

    :param episode_batch: (dict) the episode batch
    :return: (int) the number of transitions in episode batch
    """
    shape = episode_batch['u'].shape
    return shape[0] * shape[1]


def reshape_for_broadcasting(source, target):
    """
    Reshapes a tensor (source) to have the correct shape and dtype of the target before broadcasting it with MPI.

    :param source: (TensorFlow Tensor) the input tensor
    :param target: (TensorFlow Tensor) the target tensor
    :return: (TensorFlow Tensor) the rehshaped tensor
    """
    dim = len(target.get_shape())
    shape = ([1] * (dim - 1)) + [-1]
    return tf.reshape(tf.cast(source, target.dtype), shape)
