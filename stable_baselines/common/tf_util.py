import copy
import os
import functools
import collections
import multiprocessing

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

from stable_baselines import logger


def switch(condition, then_expression, else_expression):
    """
    Switches between two operations depending on a scalar value (int or bool).
    Note that both `then_expression` and `else_expression`
    should be symbolic tensors of the *same shape*.

    :param condition: (TensorFlow Tensor) scalar tensor.
    :param then_expression: (TensorFlow Operation)
    :param else_expression: (TensorFlow Operation)
    :return: (TensorFlow Operation) the switch output
    """
    x_shape = copy.copy(then_expression.get_shape())
    out_tensor = tf.cond(tf.cast(condition, 'bool'),
                         lambda: then_expression,
                         lambda: else_expression)
    out_tensor.set_shape(x_shape)
    return out_tensor


# ================================================================
# Extras
# ================================================================

def leaky_relu(tensor, leak=0.2):
    """
    Leaky ReLU
    http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf

    :param tensor: (float) the input value
    :param leak: (float) the leaking coeficient when the function is saturated
    :return: (float) Leaky ReLU output
    """
    f_1 = 0.5 * (1 + leak)
    f_2 = 0.5 * (1 - leak)
    return f_1 * tensor + f_2 * abs(tensor)


# ================================================================
# Mathematical utils
# ================================================================

def huber_loss(tensor, delta=1.0):
    """
    Reference: https://en.wikipedia.org/wiki/Huber_loss

    :param tensor: (TensorFlow Tensor) the input value
    :param delta: (float) huber loss delta value
    :return: (TensorFlow Tensor) huber loss output
    """
    return tf.where(
        tf.abs(tensor) < delta,
        tf.square(tensor) * 0.5,
        delta * (tf.abs(tensor) - 0.5 * delta)
    )


# ================================================================
# Global session
# ================================================================

def make_session(num_cpu=None, make_default=False, graph=None):
    """
    Returns a session that will use <num_cpu> CPU's only

    :param num_cpu: (int) number of CPUs to use for TensorFlow
    :param make_default: (bool) if this should return an InteractiveSession or a normal Session
    :param graph: (TensorFlow Graph) the graph of the session
    :return: (TensorFlow session)
    """
    if num_cpu is None:
        num_cpu = int(os.getenv('RCALL_NUM_CPU', multiprocessing.cpu_count()))
    tf_config = tf.ConfigProto(
        allow_soft_placement=True,
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu)
    # Prevent tensorflow from taking all the gpu memory
    tf_config.gpu_options.allow_growth = True
    if make_default:
        return tf.InteractiveSession(config=tf_config, graph=graph)
    else:
        return tf.Session(config=tf_config, graph=graph)


def single_threaded_session(make_default=False, graph=None):
    """
    Returns a session which will only use a single CPU

    :param make_default: (bool) if this should return an InteractiveSession or a normal Session
    :param graph: (TensorFlow Graph) the graph of the session
    :return: (TensorFlow session)
    """
    return make_session(num_cpu=1, make_default=make_default, graph=graph)


def in_session(func):
    """
    wrappes a function so that it is in a TensorFlow Session

    :param func: (function) the function to wrap
    :return: (function)
    """

    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        with tf.Session():
            func(*args, **kwargs)

    return newfunc


ALREADY_INITIALIZED = set()


def initialize(sess=None):
    """
    Initialize all the uninitialized variables in the global scope.

    :param sess: (TensorFlow Session)
    """
    if sess is None:
        sess = tf.get_default_session()
    new_variables = set(tf.global_variables()) - ALREADY_INITIALIZED
    sess.run(tf.variables_initializer(new_variables))
    ALREADY_INITIALIZED.update(new_variables)


# ================================================================
# Model components
# ================================================================

def normc_initializer(std=1.0, axis=0):
    """
    Return a parameter initializer for TensorFlow

    :param std: (float) standard deviation
    :param axis: (int) the axis to normalize on
    :return: (function)
    """

    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=axis, keepdims=True))
        return tf.constant(out)

    return _initializer


def conv2d(input_tensor, num_filters, name, filter_size=(3, 3), stride=(1, 1),
           pad="SAME", dtype=tf.float32, collections=None, summary_tag=None):
    """
    Creates a 2d convolutional layer for TensorFlow

    :param input_tensor: (TensorFlow Tensor) The input tensor for the convolution
    :param num_filters: (int) The number of filters
    :param name: (str) The TensorFlow variable scope
    :param filter_size: (tuple) The filter size
    :param stride: (tuple) The stride of the convolution
    :param pad: (str) The padding type ('VALID' or 'SAME')
    :param dtype: (type) The data type for the Tensors
    :param collections: (list) List of graph collections keys to add the Variable to
    :param summary_tag: (str) image summary name, can be None for no image summary
    :return: (TensorFlow Tensor) 2d convolutional layer
    """
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(input_tensor.get_shape()[3]), num_filters]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = intprod(filter_shape[:3])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = intprod(filter_shape[:2]) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        weight = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                                 collections=collections)
        bias = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.zeros_initializer(),
                               collections=collections)

        if summary_tag is not None:
            tf.summary.image(summary_tag,
                             tf.transpose(tf.reshape(weight, [filter_size[0], filter_size[1], -1, 1]), [2, 0, 1, 3]),
                             max_outputs=10)

        return tf.nn.conv2d(input_tensor, weight, stride_shape, pad) + bias


# ================================================================
# Theano-like Function
# ================================================================

def function(inputs, outputs, updates=None, givens=None):
    """
    Just like Theano function. Take a bunch of tensorflow placeholders and expressions
    computed based on those placeholders and produces f(inputs) -> outputs. Function f takes
    values to be fed to the input's placeholders and produces the values of the expressions
    in outputs.

    Input values can be passed in the same order as inputs or can be provided as kwargs based
    on placeholder name (passed to constructor or accessible via placeholder.op.name).

    Example:
        x = tf.placeholder(tf.int32, (), name="x")
        y = tf.placeholder(tf.int32, (), name="y")
        z = 3 * x + 2 * y
        lin = function([x, y], z, givens={y: 0})

        with single_threaded_session():
            initialize()

            assert lin(2) == 6
            assert lin(x=3) == 9
            assert lin(2, 2) == 10
            assert lin(x=2, y=3) == 12

    :param inputs: (TensorFlow Tensor or Object with make_feed_dict) list of input arguments
    :param outputs: (TensorFlow Tensor) list of outputs or a single output to be returned from function. Returned
        value will also have the same shape.
    :param updates: (list) update functions
    :param givens: (dict) the values known for the output
    """
    if isinstance(outputs, list):
        return _Function(inputs, outputs, updates, givens=givens)
    elif isinstance(outputs, (dict, collections.OrderedDict)):
        func = _Function(inputs, outputs.values(), updates, givens=givens)
        return lambda *args, **kwargs: type(outputs)(zip(outputs.keys(), func(*args, **kwargs)))
    else:
        func = _Function(inputs, [outputs], updates, givens=givens)
        return lambda *args, **kwargs: func(*args, **kwargs)[0]


class _Function(object):
    def __init__(self, inputs, outputs, updates, givens):
        """
        Theano like function

        :param inputs: (TensorFlow Tensor or Object with make_feed_dict) list of input arguments
        :param outputs: (TensorFlow Tensor) list of outputs or a single output to be returned from function. Returned
            value will also have the same shape.
        :param updates: (list) update functions
        :param givens: (dict) the values known for the output
        """
        for inpt in inputs:
            if not hasattr(inpt, 'make_feed_dict') and not (isinstance(inpt, tf.Tensor)and len(inpt.op.inputs) == 0):
                assert False, "inputs should all be placeholders, constants, or have a make_feed_dict method"
        self.inputs = inputs
        updates = updates or []
        self.update_group = tf.group(*updates)
        self.outputs_update = list(outputs) + [self.update_group]
        self.givens = {} if givens is None else givens

    @classmethod
    def _feed_input(cls, feed_dict, inpt, value):
        if hasattr(inpt, 'make_feed_dict'):
            feed_dict.update(inpt.make_feed_dict(value))
        else:
            feed_dict[inpt] = value

    def __call__(self, *args, sess=None, **kwargs):
        assert len(args) <= len(self.inputs), "Too many arguments provided"
        if sess is None:
            sess = tf.get_default_session()
        feed_dict = {}
        # Update the args
        for inpt, value in zip(self.inputs, args):
            self._feed_input(feed_dict, inpt, value)
        # Update feed dict with givens.
        for inpt in self.givens:
            feed_dict[inpt] = feed_dict.get(inpt, self.givens[inpt])
        results = sess.run(self.outputs_update, feed_dict=feed_dict, **kwargs)[:-1]
        return results


# ================================================================
# Flat vectors
# ================================================================

def var_shape(tensor):
    """
    get TensorFlow Tensor shape

    :param tensor: (TensorFlow Tensor) the input tensor
    :return: ([int]) the shape
    """
    out = tensor.get_shape().as_list()
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out


def numel(tensor):
    """
    get TensorFlow Tensor's number of elements

    :param tensor: (TensorFlow Tensor) the input tensor
    :return: (int) the number of elements
    """
    return intprod(var_shape(tensor))


def intprod(tensor):
    """
    calculates the product of all the elements in a list

    :param tensor: ([Number]) the list of elements
    :return: (int) the product truncated
    """
    return int(np.prod(tensor))


def flatgrad(loss, var_list, clip_norm=None):
    """
    calculates the gradient and flattens it

    :param loss: (float) the loss value
    :param var_list: ([TensorFlow Tensor]) the variables
    :param clip_norm: (float) clip the gradients (disabled if None)
    :return: ([TensorFlow Tensor]) flattend gradient
    """
    grads = tf.gradients(loss, var_list)
    if clip_norm is not None:
        grads = [tf.clip_by_norm(grad, clip_norm=clip_norm) for grad in grads]
    return tf.concat(axis=0, values=[
        tf.reshape(grad if grad is not None else tf.zeros_like(v), [numel(v)])
        for (v, grad) in zip(var_list, grads)
    ])


class SetFromFlat(object):
    def __init__(self, var_list, dtype=tf.float32, sess=None):
        """
        Set the parameters from a flat vector

        :param var_list: ([TensorFlow Tensor]) the variables
        :param dtype: (type) the type for the placeholder
        :param sess: (TensorFlow Session)
        """
        shapes = list(map(var_shape, var_list))
        total_size = np.sum([intprod(shape) for shape in shapes])

        self.theta = theta = tf.placeholder(dtype, [total_size])
        start = 0
        assigns = []
        for (shape, _var) in zip(shapes, var_list):
            size = intprod(shape)
            assigns.append(tf.assign(_var, tf.reshape(theta[start:start + size], shape)))
            start += size
        self.operation = tf.group(*assigns)
        self.sess = sess

    def __call__(self, theta):
        if self.sess is None:
            return tf.get_default_session().run(self.operation, feed_dict={self.theta: theta})
        else:
            return self.sess.run(self.operation, feed_dict={self.theta: theta})


class GetFlat(object):
    def __init__(self, var_list, sess=None):
        """
        Get the parameters as a flat vector

        :param var_list: ([TensorFlow Tensor]) the variables
        :param sess: (TensorFlow Session)
        """
        self.operation = tf.concat(axis=0, values=[tf.reshape(v, [numel(v)]) for v in var_list])
        self.sess = sess

    def __call__(self):
        if self.sess is None:
            return tf.get_default_session().run(self.operation)
        else:
            return self.sess.run(self.operation)


def flattenallbut0(tensor):
    """
    flatten all the dimension, except from the first one

    :param tensor: (TensorFlow Tensor) the input tensor
    :return: (TensorFlow Tensor) the flattened tensor
    """
    return tf.reshape(tensor, [-1, intprod(tensor.get_shape().as_list()[1:])])


# ================================================================
# Diagnostics
# ================================================================

def display_var_info(_vars):
    """
    log variable information, for debug purposes

    :param _vars: ([TensorFlow Tensor]) the variables
    """
    count_params = 0
    for _var in _vars:
        name = _var.name
        if "/Adam" in name or "beta1_power" in name or "beta2_power" in name:
            continue
        v_params = np.prod(_var.shape.as_list())
        count_params += v_params
        if "/b:" in name or "/biases" in name:
            continue  # Wx+b, bias is not interesting to look at => count params, but not print
        logger.info("   %s%s %i params %s" % (name, " " * (55 - len(name)), v_params, str(_var.shape)))

    logger.info("Total model parameters: %0.2f million" % (count_params * 1e-6))


def get_available_gpus():
    """
    Return a list of all the available GPUs

    :return: ([str]) the GPUs available
    """
    # recipe from here:
    # https://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


# ================================================================
# Saving variables
# ================================================================

def load_state(fname, sess=None, var_list=None):
    """
    Load a TensorFlow saved model

    :param fname: (str) the graph name
    :param sess: (TensorFlow Session) the session, if None: get_default_session()
    :param var_list: ([TensorFlow Tensor] or dict(str: TensorFlow Tensor)) A list of Variable/SaveableObject,
        or a dictionary mapping names to SaveableObject`s. If ``None``, defaults to the list of all saveable objects.
    """
    if sess is None:
        sess = tf.get_default_session()

    # avoir crashing when loading the direct name without explicitly adding the root folder
    if os.path.dirname(fname) == '':
        fname = os.path.join('./', fname)

    saver = tf.train.Saver(var_list=var_list)
    saver.restore(sess, fname)


def save_state(fname, sess=None, var_list=None):
    """
    Save a TensorFlow model

    :param fname: (str) the graph name
    :param sess: (TensorFlow Session) The tf session, if None, get_default_session()
    :param var_list: ([TensorFlow Tensor] or dict(str: TensorFlow Tensor)) A list of Variable/SaveableObject,
        or a dictionary mapping names to SaveableObject`s. If ``None``, defaults to the list of all saveable objects.
    """
    if sess is None:
        sess = tf.get_default_session()

    dir_name = os.path.dirname(fname)
    # avoir crashing when saving the direct name without explicitly adding the root folder
    if dir_name == '':
        dir_name = './'
        fname = os.path.join(dir_name, fname)
    os.makedirs(dir_name, exist_ok=True)

    saver = tf.train.Saver(var_list=var_list)
    saver.save(sess, fname)


# ================================================================
# retrieving variables
# ================================================================

def get_trainable_vars(name):
    """
    returns the trainable variables

    :param name: (str) the scope
    :return: ([TensorFlow Variable])
    """
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)


def get_globals_vars(name):
    """
    returns the trainable variables

    :param name: (str) the scope
    :return: ([TensorFlow Variable])
    """
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)


def outer_scope_getter(scope, new_scope=""):
    """
    remove a scope layer for the getter

    :param scope: (str) the layer to remove
    :param new_scope: (str) optional replacement name
    :return: (function (function, str, ``*args``, ``**kwargs``): Tensorflow Tensor)
    """
    def _getter(getter, name, *args, **kwargs):
        name = name.replace(scope + "/", new_scope, 1)
        val = getter(name, *args, **kwargs)
        return val
    return _getter
