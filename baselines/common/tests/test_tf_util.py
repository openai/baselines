# tests for tf_util
import tensorflow as tf
from baselines.common.tf_util import (
    function,
    initialize,
    single_threaded_session
)


def test_function():
    with tf.Graph().as_default():
        x = tf.placeholder(tf.int32, (), name="x")
        y = tf.placeholder(tf.int32, (), name="y")
        z = 3 * x + 2 * y
        lin = function([x, y], z, givens={y: 0})

        with single_threaded_session():
            initialize()

            assert lin(2) == 6
            assert lin(2, 2) == 10


def test_multikwargs():
    with tf.Graph().as_default():
        x = tf.placeholder(tf.int32, (), name="x")
        with tf.variable_scope("other"):
            x2 = tf.placeholder(tf.int32, (), name="x")
        z = 3 * x + 2 * x2

        lin = function([x, x2], z, givens={x2: 0})
        with single_threaded_session():
            initialize()
            assert lin(2) == 6
            assert lin(2, 2) == 10


if __name__ == '__main__':
    test_function()
    test_multikwargs()
