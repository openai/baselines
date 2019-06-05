import gym
import numpy as np
import tensorflow as tf

from stable_baselines.a2c.utils import conv
from stable_baselines.common.input import observation_input


ENV_ID = 'BreakoutNoFrameskip-v4'
SEED = 3


def test_conv_kernel():
    """Test convolution kernel with various input formats."""
    filter_size_1 = 4   # The size of squared filter for the first layer
    filter_size_2 = (3, 5)  # The size of non-squared filter for the second layer
    target_shape_1 = [2, 52, 40, 32]  # The desired shape of the first layer
    target_shape_2 = [2, 13, 9, 32]  # The desired shape of the second layer
    kwargs = {}
    n_envs = 1
    n_steps = 2
    n_batch = n_envs * n_steps
    scale = False
    env = gym.make(ENV_ID)
    ob_space = env.observation_space

    with tf.Graph().as_default():
        _, scaled_images = observation_input(ob_space, n_batch, scale=scale)
        activ = tf.nn.relu
        layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=filter_size_1, stride=4
                             , init_scale=np.sqrt(2), **kwargs))
        layer_2 = activ(conv(layer_1, 'c2', n_filters=32, filter_size=filter_size_2, stride=4
                             , init_scale=np.sqrt(2), **kwargs))
        assert layer_1.shape == target_shape_1 \
            , "The shape of layer based on the squared kernel matrix is not correct. " \
              "The current shape is {} and the desired shape is {}".format(layer_1.shape, target_shape_1)
        assert layer_2.shape == target_shape_2 \
            , "The shape of layer based on the non-squared kernel matrix is not correct. " \
              "The current shape is {} and the desired shape is {}".format(layer_2.shape, target_shape_2)
    env.close()
