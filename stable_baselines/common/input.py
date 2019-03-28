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
        observation_ph = tf.placeholder(shape=(batch_size,), dtype=tf.int32, name=name)
        processed_observations = tf.cast(tf.one_hot(observation_ph, ob_space.n), tf.float32)
        return observation_ph, processed_observations

    elif isinstance(ob_space, Box):
        observation_ph = tf.placeholder(shape=(batch_size,) + ob_space.shape, dtype=ob_space.dtype, name=name)
        processed_observations = tf.cast(observation_ph, tf.float32)
        # rescale to [1, 0] if the bounds are defined
        if (scale and
           not np.any(np.isinf(ob_space.low)) and not np.any(np.isinf(ob_space.high)) and
           np.any((ob_space.high - ob_space.low) != 0)):

            # equivalent to processed_observations / 255.0 when bounds are set to [255, 0]
            processed_observations = ((processed_observations - ob_space.low) / (ob_space.high - ob_space.low))
        return observation_ph, processed_observations

    elif isinstance(ob_space, MultiBinary):
        observation_ph = tf.placeholder(shape=(batch_size, ob_space.n), dtype=tf.int32, name=name)
        processed_observations = tf.cast(observation_ph, tf.float32)
        return observation_ph, processed_observations

    elif isinstance(ob_space, MultiDiscrete):
        observation_ph = tf.placeholder(shape=(batch_size, len(ob_space.nvec)), dtype=tf.int32, name=name)
        processed_observations = tf.concat([
            tf.cast(tf.one_hot(input_split, ob_space.nvec[i]), tf.float32) for i, input_split
            in enumerate(tf.split(observation_ph, len(ob_space.nvec), axis=-1))
        ], axis=-1)
        return observation_ph, processed_observations

    else:
        raise NotImplementedError("Error: the model does not support input space of type {}".format(
            type(ob_space).__name__))
