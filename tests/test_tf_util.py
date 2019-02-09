# tests for tf_util
import numpy as np
import tensorflow as tf

from stable_baselines.common.tf_util import function, initialize, single_threaded_session, is_image


def test_function():
    """
    test the function function in tf_util
    """
    with tf.Graph().as_default():
        x_ph = tf.placeholder(tf.int32, (), name="x")
        y_ph = tf.placeholder(tf.int32, (), name="y")
        z_ph = 3 * x_ph + 2 * y_ph
        linear_fn = function([x_ph, y_ph], z_ph, givens={y_ph: 0})

        with single_threaded_session():
            initialize()

            assert linear_fn(2) == 6
            assert linear_fn(2, 2) == 10


def test_multikwargs():
    """
    test the function function in tf_util
    """
    with tf.Graph().as_default():
        x_ph = tf.placeholder(tf.int32, (), name="x")
        with tf.variable_scope("other"):
            x2_ph = tf.placeholder(tf.int32, (), name="x")
        z_ph = 3 * x_ph + 2 * x2_ph

        linear_fn = function([x_ph, x2_ph], z_ph, givens={x2_ph: 0})
        with single_threaded_session():
            initialize()
            assert linear_fn(2) == 6
            assert linear_fn(2, 2) == 10


def test_image_detection():
    rgb = (32, 64, 3)
    gray = (43, 23, 1)
    rgbd = (12, 32, 4)
    invalid_1 = (32, 12)
    invalid_2 = (12, 32, 6)

    # TF checks
    for shape in (rgb, gray, rgbd):
        assert is_image(tf.placeholder(tf.uint8, shape=shape))

    for shape in (invalid_1, invalid_2):
        assert not is_image(tf.placeholder(tf.uint8, shape=shape))

    # Numpy checks
    for shape in (rgb, gray, rgbd):
        assert is_image(np.ones(shape))

    for shape in (invalid_1, invalid_2):
        assert not is_image(np.ones(shape))
