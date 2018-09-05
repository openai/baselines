import numpy as np
import tensorflow as tf
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete


def observation_input(ob_space, batch_size=None, name='Ob', scale=False):
    """
    Build observation input with encoding depending on the observation space type

    When using Box ob_space, the input will be normalized between [1, 0] on the bounds ob_space.low and ob_space.high.

    :param ob_space: (Gym Space) The observation space
    :param batch_size: (int) batch size for input
                       (default is None, so that resulting input placeholder can take tensors with any batch size)
    :param name: (str) tensorflow variable name for input placeholder
    :param scale: (bool) whether or not to scale the input
    :return: (TensorFlow Tensor, TensorFlow Tensor) input_placeholder, processed_input_tensor
    """
    if isinstance(ob_space, Discrete):
        input_x = tf.placeholder(shape=(batch_size,), dtype=tf.int32, name=name)
        processed_x = tf.to_float(tf.one_hot(input_x, ob_space.n))
        return input_x, processed_x

    elif isinstance(ob_space, Box):
        input_x = tf.placeholder(shape=(batch_size,) + ob_space.shape, dtype=ob_space.dtype, name=name)
        processed_x = tf.to_float(input_x)
        # rescale to [1, 0] if the bounds are defined
        if (scale and
           not np.any(np.isinf(ob_space.low)) and not np.any(np.isinf(ob_space.high)) and
           np.any((ob_space.high - ob_space.low) != 0)):

            # equivalent to processed_x / 255.0 when bounds are set to [255, 0]
            processed_x = ((processed_x - ob_space.low) / (ob_space.high - ob_space.low))
        return input_x, processed_x

    elif isinstance(ob_space, MultiBinary):
        input_x = tf.placeholder(shape=(batch_size, ob_space.n), dtype=tf.int32, name=name)
        processed_x = tf.to_float(input_x)
        return input_x, processed_x

    elif isinstance(ob_space, MultiDiscrete):
        input_x = tf.placeholder(shape=(batch_size, len(ob_space.nvec)), dtype=tf.int32, name=name)
        processed_x = tf.concat([tf.to_float(tf.one_hot(input_split, ob_space.nvec[i]))
                                 for i, input_split in enumerate(tf.split(input_x, len(ob_space.nvec), axis=-1))],
                                axis=-1)
        return input_x, processed_x

    else:
        raise NotImplementedError("Error: the model does not support input space of type {}".format(
            type(ob_space).__name__))
