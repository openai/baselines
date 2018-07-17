# tests for tf_util
import tensorflow as tf

from stable_baselines.common.tf_util import function, initialize, single_threaded_session


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


if __name__ == '__main__':
    test_function()
    test_multikwargs()
