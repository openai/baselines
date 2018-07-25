import tensorflow as tf
from gym.spaces import Discrete, Box


def observation_input(ob_space, batch_size=None, name='Ob', n_stack=None):
    """
    Build observation input with encoding depending on the observation space type

    :param ob_space: (Gym Space) The observation space
    :param batch_size: (int) batch size for input
                       (default is None, so that resulting input placeholder can take tensors with any batch size)
    :param name: (str) tensorflow variable name for input placeholder
    :param n_stack: (int) the number of frames to stack (None for no stacking)
    :return: (TensorFlow Tensor, TensorFlow Tensor) input_placeholder, processed_input_tensor
    """
    if isinstance(ob_space, Discrete):
        if n_stack is None:
            input_x = tf.placeholder(shape=(batch_size,), dtype=tf.int32, name=name)
            processed_x = tf.to_float(tf.one_hot(input_x, ob_space.n))
        else:
            input_x = tf.placeholder(shape=(batch_size, n_stack), dtype=tf.int32, name=name)
            # stack every one_hot encoding end to end
            processed_x = tf.to_float(
                tf.reshape(tf.stack([tf.one_hot(input_x[:, i], ob_space.n) for i in range(n_stack)], axis=1),
                           [-1, ob_space.n * n_stack])
            )
        return input_x, processed_x

    elif isinstance(ob_space, Box):
        if n_stack is None:
            input_shape = (batch_size,) + ob_space.shape
        else:
            input_shape = (batch_size,) + ob_space.shape[:-1] + (ob_space.shape[-1] * n_stack,)
        input_x = tf.placeholder(shape=input_shape, dtype=ob_space.dtype, name=name)
        processed_x = tf.to_float(input_x)
        return input_x, processed_x

    else:
        raise NotImplementedError("Error: the model does not support input space of type {}".format(
            type(ob_space).__name__))
