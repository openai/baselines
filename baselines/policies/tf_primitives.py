import os
import copy
import functools
import collections
import numpy as np
import multiprocessing
import tensorflow as tf


####################################################################
# Global Variables                                                 #
####################################################################
_PLACEHOLDER_CACHE = {}  # name -> (placeholder, dtype, shape)
ALREADY_INITIALIZED = set()


#######################################################################
# TfUtil Class: Notice make sure whenever the arguments are inputs or #
# function calls to enclose them inside quotes, single for inputs and #
# double for functions                                                #
# exposes primitive tf operations                                     #
#######################################################################
class TfUtil(object):

    def __init__(self, namespace='layers'):
        self.args = None
        self.expression = None
        self.namespace = namespace
        # self.ALREADY_INITIALIZED = set()
        # self._PLACEHOLDER_CACHE = {}  # name -> (placeholder, dtype, shape)

    def convolution(self, **kwargs):
        if self.sanity_check('func_name', **kwargs):
            func_name = kwargs.pop('func_name')
            if kwargs:
                return self.get_attribute(func_name)(**kwargs)
            else:
                print("You have to pass arguments to tf.{}.{}".format(
                    self.namespace, func_name)
                )
                return
        else:
            return

    def pooling(self, **kwargs):
        if self.sanity_check('func_name', **kwargs):
            func_name = kwargs.pop('func_name')
            if kwargs:
                return self.get_attribute(func_name)(**kwargs)
            else:
                return

    def activation(self, func_name='relu'):
        self.expression = 'tf.nn.{}'.format(func_name)

        return self.eval_expression(self.expression)

    def dense(self, **kwargs):
        if kwargs:
            return self.get_attribute('dense')(**kwargs)
        else:
            print("No arguments provided for dense function!")

    def flatten(self, **kwargs):
        if kwargs:
            return self.get_attribute('flatten')(**kwargs)
        else:
            print("No arguments provided for flatten function!")

    def eval_expression(self, expression):
        try:
            return eval("{}".format(expression))
        except Exception as err:
            print("{}".format(err))

    def sanity_check(self, key, **kwargs):
        if key not in kwargs.keys():
            print("Argument missing from the parameters {}".format(key))
            return 0
        else:
            return 1

    def construct_args(self, **kwargs):
        self.args = ', '.join('{}={}'.format(k, v) for k, v in kwargs.items())

    def construct_expression(self, func_name):
            self.expression = 'tf.{}.{}({})'.format(
                self.namespace, func_name, self.args
                )

    def get_attribute(self, func_name):
        return getattr(
            self.eval_expression("tf.{}".format(self.namespace)),
            func_name
        )

    def switch(self, condition, then_expression, else_expression):
        """
        Switches between two operations depending on a scalar value
        (int or bool). Note that both `then_expression` and
        `else_expression` should be symbolic tensors of the *same
        shape*.

        # Arguments
            condition: scalar tensor.
            then_expression: TensorFlow operation.
            else_expression: TensorFlow operation.
        """
        x_shape = copy.copy(then_expression.get_shape())
        x = tf.cond(
            tf.cast(condition, 'bool'),
            lambda: then_expression,
            lambda: else_expression
        )
        x.set_shape(x_shape)
        return x

    def get_placeholder(self, name, dtype, shape):
        # global _PLACEHOLDER_CACHE
        if name in _PLACEHOLDER_CACHE:
            out, dtype1, shape1 = _PLACEHOLDER_CACHE[name]
            assert dtype1 == dtype and shape1 == shape
            return out
        else:
            out = tf.placeholder(dtype=dtype, shape=shape, name=name)
            _PLACEHOLDER_CACHE[name] = (out, dtype, shape)
            return out

    def get_placeholder_cached(self, name):
        # global _PLACEHOLDER_CACHE
        return _PLACEHOLDER_CACHE[name][0]

    ################################################################
    # Flatten vectors                                              #
    ################################################################
    def var_shape(self, x):
        out = x.get_shape().as_list()
        assert all(isinstance(a, int) for a in out), \
            "shape function assumes that shape is fully known"
        return out

    def numel(self, x):
        return self.intprod(self.var_shape(x))

    def intprod(self, x):
        return int(np.prod(x))

    def flatten_except_first(self, x):
        return tf.reshape(x, [-1, self.intprod(x.get_shape().as_list()[1:])])

    def flatten_gradients(self, loss, var_list, clip_norm=None):
        grads = tf.gradients(loss, var_list)
        if clip_norm is not None:
            grads = [
                tf.clip_by_norm(grad, clip_norm=clip_norm) for grad in grads
            ]
        return tf.concat(
            axis=0,
            values=[
                tf.reshape(
                    grad if grad is not None else tf.zeros_like(v),
                    [self.numel(v)]
                    ) for (v, grad) in zip(var_list, grads)
            ]
        )

    ################################################################
    # Global session                                               #
    ################################################################
    def init_session(self, num_cpu=None, make_default=False, graph=None):
        """Returns a session that will use <num_cpu> CPU's only"""
        if num_cpu is None:
            num_cpu = int(os.getenv('RCALL_NUM_CPU', multiprocessing.cpu_count()))
        tf_config = tf.ConfigProto(
            allow_soft_placement=True,
            intra_op_parallelism_threads=num_cpu,
            inter_op_parallelism_threads=num_cpu
        )
            # device_count={'CPU': 1, 'GPU': 0},
            # log_device_placement=False,
        tf_config.gpu_options.allow_growth = True
        if make_default:
            return tf.InteractiveSession(config=tf_config, graph=graph)
        else:
            return tf.Session(config=tf_config, graph=graph)

    def single_threaded_session(self):
        """Returns a session which will only use a single CPU"""
        return self.init_session(num_cpu=1)

    def in_session(f):
        @functools.wraps(f)
        def newfunc(*args, **kwargs):
            tf_config = tf.ConfigProto(
                device_count={'CPU': 1, 'GPU': 0},
                allow_soft_placement=True,
                log_device_placement=False
            )
            with tf.Session(config=tf_config):
                f(*args, **kwargs)
        return newfunc

    def init_vars(self):
        """Initialize all the uninitialized variables in the global scope."""
        new_variables = set(tf.global_variables()) - ALREADY_INITIALIZED
        tf.get_default_session().run(
            tf.variables_initializer(new_variables)
        )
        ALREADY_INITIALIZED.update(new_variables)

    ################################################################
    # Saving variables                                             #
    ################################################################
    def load_state(self, fname):
        saver = tf.train.Saver()
        return saver.restore(tf.get_default_session(), fname)

    def save_state(self, fname):
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        saver = tf.train.Saver()
        saver.save(tf.get_default_session(), fname)

    def reset_graph_and_vars():
        _PLACEHOLDER_CACHE = {}
        ALREADY_INITIALIZED = {}
        tf.reset_default_graph()

    ################################################################
    # Diagnostics                                                  #
    ################################################################
    def display_var_info(self, variables):
        from utils import logger
        count_params = 0
        for v in variables:
            name = v.name
            if "/Adam" in name or "beta1_power" in name or "beta2_power" in name:
                continue
            v_params = np.prod(v.shape.as_list())
            count_params += v_params
            if "/b:" in name or "/biases" in name: continue    # Wx+b, bias is not
            # interesting to look at => count params, but not print
            logger.info("   %s%s %i params %s" % (
                name,
                " " * (55 - len(name)),
                v_params,
                str(v.shape))
            )

        logger.info("Total model parameters: %0.2f million"
                    % (count_params * 1e-6))

    def get_available_gpus(self):
        # recipe from here:
        # https://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa

        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']

    ################################################################
    # Theano-like function                                         #
    ################################################################
    def function(self, inputs, outputs, updates=None, givens=None):
        """
        Just like Theano function. Take a bunch of tensorflow
           placeholders and expressions computed based on those
           placeholders and produces f(inputs) -> outputs.  Function f
           takes values to be fed to the input's placeholders and
           produces the values of the expressions in outputs.

        Input values can be passed in the same order as inputs or can
        be provided as kwargs based on placeholder name (passed to
        constructor or accessible via placeholder.op.name).

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

        Parameters
        ----------
        inputs: [tf.placeholder, tf.constant, or
                 object with make_feed_dict method] list of input arguments
        outputs: [tf.Variable] or tf.Variable
            list of outputs or a single output to be returned from function.
            Returned value will also have the same shape.
        """
        if isinstance(outputs, list):
            return _Function(inputs, outputs, updates, givens=givens)
        elif isinstance(outputs, (dict, collections.OrderedDict)):
            f = _Function(inputs, outputs.values(), updates, givens=givens)
            return lambda *args, **kwargs: type(outputs)(
                zip(outputs.keys(), f(*args, **kwargs))
            )
        else:
            f = _Function(inputs, [outputs], updates, givens=givens)
            return lambda *args, **kwargs: f(*args, **kwargs)[0]


####################################################################
# Theano-like Function                                             #
####################################################################
class _Function(object):
    def __init__(self, inputs, outputs, updates, givens):
        for inpt in inputs:
            if not hasattr(inpt, 'make_feed_dict') and \
               not (type(inpt) is tf.Tensor and len(inpt.op.inputs) == 0):
                assert False, "inputs should all be placeholders,"
                " constants, or have a make_feed_dict method"
        self.inputs = inputs
        updates = updates or []
        self.update_group = tf.group(*updates)
        self.outputs_update = list(outputs) + [self.update_group]
        self.givens = {} if givens is None else givens

    def _feed_input(self, feed_dict, inpt, value):
        if hasattr(inpt, 'make_feed_dict'):
            feed_dict.update(inpt.make_feed_dict(value))
        else:
            feed_dict[inpt] = value

    def __call__(self, *args):
        assert len(args) <= len(self.inputs), "Too many arguments provided"
        feed_dict = {}
        # Update the args
        for inpt, value in zip(self.inputs, args):
            self._feed_input(feed_dict, inpt, value)
        # Update feed dict with givens.
        for inpt in self.givens:
            feed_dict[inpt] = feed_dict.get(inpt, self.givens[inpt])
        # tf.get_default_session() only works with InteractiveSession
        # which inherently has problems. If it is not closed properly
        # memory hogs up
        # results = tf.get_default_session().run(
        results = tf.get_default_session().run(
            self.outputs_update,
            feed_dict=feed_dict
        )[:-1]
        return results


class SetFromFlat(TfUtil):
    def __init__(self, var_list, dtype=tf.float32):
        assigns = []
        shapes = list(map(self.var_shape, var_list))
        total_size = np.sum([self.intprod(shape) for shape in shapes])

        self.theta = theta = tf.placeholder(dtype, [total_size])
        start = 0
        assigns = []
        for (shape, v) in zip(shapes, var_list):
            size = self.intprod(shape)
            assigns.append(tf.assign(v, tf.reshape(theta[start:start + size],
                                                   shape)))
            start += size
        self.op = tf.group(*assigns)

    def __call__(self, theta):
        tf.get_default_session().run(self.op, feed_dict={self.theta: theta})


class GetFlat(TfUtil):
    def __init__(self, var_list):
        self.op = tf.concat(axis=0, values=[
            tf.reshape(v, [self.numel(v)]) for v in var_list])

    def __call__(self):
        return tf.get_default_session().run(self.op)
