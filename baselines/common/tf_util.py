import numpy as np
import tensorflow as tf  # pylint: ignore-module
import copy

def switch(condition, then_expression, else_expression):
    """Switches between two operations depending on a scalar value (int or bool).
    Note that both `then_expression` and `else_expression`
    should be symbolic tensors of the *same shape*.

    # Arguments
        condition: scalar tensor.
        then_expression: TensorFlow operation.
        else_expression: TensorFlow operation.
    """
    x_shape = copy.copy(then_expression.get_shape())
    x = tf.cond(tf.cast(condition, 'bool'),
                lambda: then_expression,
                lambda: else_expression)
    x.set_shape(x_shape)
    return x

# ================================================================
# Extras
# ================================================================

def lrelu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)

# ================================================================
# Mathematical utils
# ================================================================

def huber_loss(x, delta=1.0):
    """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta)
    )

# ================================================================
# Model components
# ================================================================

def normc_initializer(std=1.0, axis=0):
    def _initializer(shape, dtype=None, partition_info=None):  # pylint: disable=W0613
        out = np.random.randn(*shape).astype(dtype.as_numpy_dtype)
        out *= std / np.sqrt(np.square(out).sum(axis=axis, keepdims=True))
        return tf.constant(out)
    return _initializer

def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None,
           summary_tag=None):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = intprod(filter_shape[:3])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = intprod(filter_shape[:2]) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.zeros_initializer(),
                            collections=collections)

        if summary_tag is not None:
            tf.summary.image(summary_tag,
                             tf.transpose(tf.reshape(w, [filter_size[0], filter_size[1], -1, 1]),
                                          [2, 0, 1, 3]),
                             max_images=10)

        return tf.nn.conv2d(x, w, stride_shape, pad) + b

# ================================================================
# Flat vectors
# ================================================================

def var_shape(x):
    out = x.get_shape().as_list()
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out

def numel(x):
    return intprod(var_shape(x))

def intprod(x):
    return int(np.prod(x))

def flatgrad(grads, var_list, clip_norm=None):
    if clip_norm is not None:
        grads = [tf.clip_by_norm(grad, clip_norm=clip_norm) for grad in grads]
    return tf.concat(axis=0, values=[
        tf.reshape(grad if grad is not None else tf.zeros_like(v), [numel(v)])
        for (v, grad) in zip(var_list, grads)
    ])

class SetFromFlat(object):
    def __init__(self, var_list, dtype=tf.float32):
        self.shapes = list(map(var_shape, var_list))
        self.total_size = np.sum([intprod(shape) for shape in self.shapes])
        self.var_list = var_list

    def __call__(self, theta):
        start = 0
        for (shape, v) in zip(self.shapes, self.var_list):
            size = intprod(shape)
            v.assign(tf.reshape(theta[start:start + size], shape))
            start += size

class GetFlat(object):
    def __init__(self, var_list):
        self.var_list = var_list

    def __call__(self):
        return tf.concat(axis=0, values=[tf.reshape(v, [numel(v)]) for v in self.var_list]).numpy()

def flattenallbut0(x):
    return tf.reshape(x, [-1, intprod(x.get_shape().as_list()[1:])])


# ================================================================
# Shape adjustment for feeding into tf tensors
# ================================================================
def adjust_shape(input_tensor, data):
    '''
    adjust shape of the data to the shape of the tensor if possible.
    If shape is incompatible, AssertionError is thrown

    Parameters:
        input_tensor    tensorflow input tensor

        data            input data to be (potentially) reshaped to be fed into input

    Returns:
        reshaped data
    '''

    if not isinstance(data, np.ndarray) and not isinstance(data, list):
        return data
    if isinstance(data, list):
        data = np.array(data)

    input_shape = [x or -1 for x in input_tensor.shape.as_list()]

    assert _check_shape(input_shape, data.shape), \
        'Shape of data {} is not compatible with shape of the input {}'.format(data.shape, input_shape)

    return np.reshape(data, input_shape)


def _check_shape(input_shape, data_shape):
    ''' check if two shapes are compatible (i.e. differ only by dimensions of size 1, or by the batch dimension)'''

    squeezed_input_shape = _squeeze_shape(input_shape)
    squeezed_data_shape = _squeeze_shape(data_shape)

    for i, s_data in enumerate(squeezed_data_shape):
        s_input = squeezed_input_shape[i]
        if s_input != -1 and s_data != s_input:
            return False

    return True


def _squeeze_shape(shape):
    return [x for x in shape if x != 1]

# ================================================================
# Tensorboard interfacing
# ================================================================

def launch_tensorboard_in_background(log_dir):
    '''
    To log the Tensorflow graph when using rl-algs
    algorithms, you can run the following code
    in your main script:
        import threading, time
        def start_tensorboard(session):
            time.sleep(10) # Wait until graph is setup
            tb_path = osp.join(logger.get_dir(), 'tb')
            summary_writer = tf.summary.FileWriter(tb_path, graph=session.graph)
            summary_op = tf.summary.merge_all()
            launch_tensorboard_in_background(tb_path)
        session = tf.get_default_session()
        t = threading.Thread(target=start_tensorboard, args=([session]))
        t.start()
    '''
    import subprocess
    subprocess.Popen(['tensorboard', '--logdir', log_dir])
