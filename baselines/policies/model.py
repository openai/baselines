# coding: utf-8

import os
import joblib
import inspect
import functools
import numpy as np
import tensorflow as tf
from policies.tf_primitives import TfUtil


class Model(TfUtil):
    def __init__(self, name=None):
        super(Model, self).__init__()
        self.name = name

    @property
    def variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=None)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=None)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars

    def wrap(function):
        """
        A decorator decorator, allowing to use the decorator to be used without
        parentheses if no arguments are provided. All arguments must be optional.
        """
        @functools.wraps(function)
        def decorator(*args, **kwargs):
            if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
                return function(args[0])
            else:
                return lambda wrapee: function(wrapee, *args, **kwargs)
        return decorator

    @wrap
    def define_scope(function, scope=None, *args, **kwargs):
        """
        A decorator for functions that define TensorFlow operations. The wrapped
        function will only be executed once. Subsequent calls to it will directly
        return the result so that operations are added to the graph only once.
        The operations added by the function live within a tf.variable_scope(). If
        this decorator is used with arguments, they will be forwarded to the
        variable scope. The scope name defaults to the name of the wrapped
        function.
        """
        attribute = '_cache_' + function.__name__
        name = scope or function.__name__

        @property
        @functools.wraps(function)
        def decorator(self):
            if not hasattr(self, attribute):
                with tf.variable_scope(name, *args, **kwargs):
                    setattr(self, attribute, function(self))
            return getattr(self, attribute)
        return decorator

    def store_args(method):
        """
        Stores provided method args as instance attributes.
        """
        argspec = inspect.getfullargspec(method)
        defaults = {}
        if argspec.defaults is not None:
            defaults = dict(
                zip(argspec.args[-len(argspec.defaults):], argspec.defaults)
            )
        if argspec.kwonlydefaults is not None:
            defaults.update(argspec.kwonlydefaults)
        arg_names = argspec.args[1:]

        @functools.wraps(method)
        def wrapper(*positional_args, **keyword_args):
            self = positional_args[0]
            # Get default arg values
            args = defaults.copy()
            # Add provided arg values
            for name, value in zip(arg_names, positional_args[1:]):
                args[name] = value
            args.update(keyword_args)
            self.__dict__.update(args)
            return method(*positional_args, **keyword_args)

        return wrapper

    def sample(self, logits):
        noise = tf.random_uniform(tf.shape(logits))
        return tf.argmax(logits - tf.log(-tf.log(noise)), 1)

    def categorical_entropy(self, logits):
        a0 = logits - tf.reduce_max(logits, 1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, 1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)

    def categorical_entropy_softmax(self, p0):
        return - tf.reduce_sum(p0 * tf.log(p0 + 1e-6), axis=1)

    def save(self, save_path, params):
        ps = self.function(inputs=[], outputs=params)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(ps, save_path)

    def load(self, load_path, params):
        loaded_params = joblib.load(load_path)
        restores = []
        for p, loaded_p in zip(params, loaded_params):
            restores.append(p.assign(loaded_p))
        self.function(inputs=[], outputs=restores)

    def ortho_init(self, scale=1.0):
        def _ortho_init(shape, dtype, partition_info=None):
            # lasagne ortho init for tf
            shape = tuple(shape)
            if len(shape) == 2:
                flat_shape = shape
            elif len(shape) == 4:  # assumes NHWC
                flat_shape = (np.prod(shape[:-1]), shape[-1])
            else:
                raise NotImplementedError
            a = np.random.normal(0.0, 1.0, flat_shape)
            u, _, v = np.linalg.svd(a, full_matrices=False)
            q = u if u.shape == flat_shape else v  # pick the one with the correct shape
            q = q.reshape(shape)
            return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
        return _ortho_init

    def conv(self, x, scope, *, nf, rf, stride, pad='VALID', init_scale=1.0,
             data_format='NHWC', one_dim_bias=False):
        if data_format == 'NHWC':
            channel_ax = 3
            strides = [1, stride, stride, 1]
            bshape = [1, 1, 1, nf]
        elif data_format == 'NCHW':
            channel_ax = 1
            strides = [1, 1, stride, stride]
            bshape = [1, nf, 1, 1]
        else:
            raise NotImplementedError

        bias_var_shape = [nf] if one_dim_bias else [1, nf, 1, 1]
        nin = x.get_shape()[channel_ax].value
        wshape = [rf, rf, nin, nf]
        with tf.variable_scope(scope):
            w = tf.get_variable("w", wshape,
                                initializer=self.ortho_init(init_scale))
            b = tf.get_variable("b", bias_var_shape,
                                initializer=tf.constant_initializer(0.0))
            if not one_dim_bias and data_format == 'NHWC':
                b = tf.reshape(b, bshape)
            return b + self.convolution(
                func_name='conv2d', input=x, filter=w, strides=strides,
                padding=pad, data_format=data_format
            )

    def fc(self, x, scope, nh, *, init_scale=1.0, init_bias=0.0):
        with tf.variable_scope(scope):
            nin = x.get_shape()[1].value
            w = tf.get_variable("w", [nin, nh],
                                initializer=self.ortho_init(init_scale))
            b = tf.get_variable("b", [nh],
                                initializer=tf.constant_initializer(init_bias))
            return tf.matmul(x, w) + b

    def batch_to_seq(self, h, nbatch, nsteps, flat=False):
        if flat:
            h = tf.reshape(h, [nbatch, nsteps])
        else:
            h = tf.reshape(h, [nbatch, nsteps, -1])
        return [
            tf.squeeze(v, [1]) for v in tf.split(
                axis=1, num_or_size_splits=nsteps, value=h
                )]

    def seq_to_batch(self, h, flat=False):
        shape = h[0].get_shape().as_list()
        if not flat:
            assert(len(shape) > 1)
            nh = h[0].get_shape()[-1].value
            return tf.reshape(tf.concat(axis=1, values=h), [-1, nh])
        else:
            return tf.reshape(tf.stack(values=h, axis=1), [-1])

    def discount_with_dones(self, rewards, dones, gamma):
        discounted = []
        r = 0
        for reward, done in zip(rewards[::-1], dones[::-1]):
            r = reward + gamma * r * (1. - done)  # fixed off by one bug
            discounted.append(r)
        return discounted[::-1]

    def get_by_index(self, x, idx):
        assert(len(x.get_shape()) == 2)
        assert(len(idx.get_shape()) == 1)
        idx_flattened = tf.range(0, x.shape[0]) * x.shape[1] + idx
        y = tf.gather(tf.reshape(x, [-1]),  # flatten input
                      idx_flattened)  # use flattened indices
        return y

    def check_shape(self, ts, shapes):
        i = 0
        for (t, shape) in zip(ts, shapes):
            assert t.get_shape().as_list() == shape, "id " + str(i) + \
                " shape " + str(t.get_shape()) + str(shape)
            i += 1

    def avg_norm(self, t):
        return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(t), axis=-1)))

    def gradient_add(self, g1, g2, param):
        print([g1, g2, param.name])
        assert (not (g1 is None and g2 is None)), param.name
        if g1 is None:
            return g2
        elif g2 is None:
            return g1
        else:
            return g1 + g2

    def q_explained_variance(self, qpred, q):
        _, vary = tf.nn.moments(q, axes=[0, 1])
        _, varpred = tf.nn.moments(q - qpred, axes=[0, 1])
        self.check_shape([vary, varpred], [[]] * 2)
        return 1.0 - (varpred / vary)

    def lstm(self, xs, ms, s, scope, nh, init_scale=1.0):
        nbatch, fan_in = [v.value for v in xs[0].get_shape()]
        nsteps = len(xs)
        with tf.variable_scope(scope):
            Wx = tf.get_variable("Wx", [fan_in, nh * 4],
                                 initializer=self.ortho_init(init_scale))
            Wh = tf.get_variable("Wh", [nh, nh * 4],
                                 initializer=self.ortho_init(init_scale))
            b = tf.get_variable("b", [nh * 4],
                                initializer=tf.constant_initializer(0.0))

        c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
        for idx, (x, m) in enumerate(zip(xs, ms)):
            c = c * (1 - m)
            h = h * (1 - m)
            z = tf.matmul(x, Wx) + tf.matmul(h, Wh) + b
            i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
            i = self.activation('sigmoid')(i)
            f = self.activation('sigmoid')(f)
            o = self.activation('sigmoid')(o)
            u = self.activation('tanh')(u)
            c = f * c + i * u
            h = o * self.activation('tanh')(c)
            xs[idx] = h
        s = tf.concat(axis=1, values=[c, h])
        return xs, s

    def nature_cnn(self, unscaled_images, **conv_kwargs):
        """
        CNN from Nature paper.
        """
        scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
        self.namespace = 'nn'
        h = self.activation('relu')(self.conv(
            scaled_images,
            'c1', nf=32,
            rf=8, stride=4,
            init_scale=np.sqrt(2),
            **conv_kwargs
            )
        )
        h2 = self.activation('relu')(self.conv(
            h,
            'c2',
            nf=64,
            rf=4,
            stride=2,
            init_scale=np.sqrt(2),
            **conv_kwargs
            )
        )
        h3 = self.activation('relu')(self.conv(
            h2,
            'c3',
            nf=64,
            rf=3,
            stride=1,
            init_scale=np.sqrt(2),
            **conv_kwargs
            )
        )
        self.namespace = 'layers'
        h3 = self.flatten(inputs=h3)

        return self.activation('relu')(self.fc(
            h3,
            'fc1',
            nh=512,
            init_scale=np.sqrt(2)
            )
        )

    def nature(self, unscaled_images, **conv_kwargs):
        """
        CNN from Nature paper.
        """
        scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
        self.namespace = 'layers'
        h = self.convolution(
            func_name='conv2d',
            inputs=scaled_images,
            filters=32,
            kernel_size=8,
            strides=4,
            activation=self.activation('relu'),
            kernel_initializer=self.ortho_init(np.sqrt(2)),
            name='c1'
            )
        h = self.convolution(
            func_name='conv2d',
            inputs=h,
            filters=64,
            kernel_size=4,
            strides=2,
            activation=self.activation('relu'),
            kernel_initializer=self.ortho_init(np.sqrt(2)),
            name='c2'
            )
        h = self.convolution(
            func_name='conv2d',
            inputs=h,
            filters=64,
            kernel_size=3,
            strides=1,
            activation=self.activation('relu'),
            kernel_initializer=self.ortho_init(np.sqrt(2)),
            name='c3'
            )
        h = self.flatten(inputs=h, name='flatten')

        return self.dense(
            inputs=h,
            units=512,
            activation=self.activation('relu'),
            kernel_initializer=self.ortho_init(np.sqrt(2)),
            name='dense'
            )

    def dens(
            self,
            x,
            size,
            name,
            weight_init=None,
            bias_init=0,
            weight_loss_dict=None,
            reuse=None
    ):
        with tf.variable_scope(name, reuse=reuse):
            assert (len(tf.get_variable_scope().name.split('/')) == 2)

            w = tf.get_variable("w", [x.get_shape()[1], size],
                                initializer=weight_init)
            b = tf.get_variable("b", [size],
                                initializer=tf.constant_initializer(bias_init))
            weight_decay_fc = 3e-4

            if weight_loss_dict is not None:
                weight_decay = tf.multiply(
                    tf.nn.l2_loss(w),
                    weight_decay_fc,
                    name='weight_decay_loss'
                )
                if weight_loss_dict is not None:
                    weight_loss_dict[w] = weight_decay_fc
                    weight_loss_dict[b] = 0.0

                tf.add_to_collection(
                    tf.get_variable_scope().name.split('/')[0] + '_' + 'losses',
                    weight_decay
                )

            return tf.nn.bias_add(tf.matmul(x, w), b)

    def kl_div(self, action_dist1, action_dist2, action_size):
        mean1, std1 = action_dist1[:, :action_size], action_dist1[:, action_size:]
        mean2, std2 = action_dist2[:, :action_size], action_dist2[:, action_size:]
        numerator = tf.square(mean1 - mean2) + tf.square(std1) - tf.square(std2)
        denominator = 2 * tf.square(std2) + 1e-8
        return tf.reduce_sum(
            numerator/denominator + tf.log(std2) - tf.log(std1),
            reduction_indices=-1
        )

    def normc_initializer(self, std=1.0, axis=0):
        def _initializer(shape, dtype=None, partition_info=None):
            out = np.random.randn(*shape).astype(np.float32)
            out *= std / np.sqrt(np.square(out).sum(axis=axis, keepdims=True))
            return tf.constant(out)
        return _initializer

    def conv2d(
            self,
            x,
            num_filters,
            name,
            filter_size=(3, 3),
            stride=(1, 1),
            pad="SAME",
            dtype=tf.float32,
            collections=None,
            summary_tag=None
    ):
        with tf.variable_scope(name):
            stride_shape = [1, stride[0], stride[1], 1]
            filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]),
                            num_filters]

            # there are "num input feature maps * filter height * filter width"
            # inputs to each hidden unit
            fan_in = self.intprod(filter_shape[:3])
            # each unit in the lower layer receives a gradient from:
            # "num output feature maps * filter height * filter width" /
            #   pooling size
            fan_out = self.intprod(filter_shape[:2]) * num_filters
            # initialize weights with random weights
            w_bound = np.sqrt(6. / (fan_in + fan_out))

            w = tf.get_variable("W", filter_shape, dtype,
                                tf.random_uniform_initializer(-w_bound, w_bound),
                                collections=collections)
            b = tf.get_variable("b", [1, 1, 1, num_filters],
                                initializer=tf.zeros_initializer(),
                                collections=collections)

            if summary_tag is not None:
                tf.summary.image(
                    summary_tag,
                    tf.transpose(
                        tf.reshape(w, [filter_size[0], filter_size[1], -1, 1]),
                        [2, 0, 1, 3]
                        ),
                    max_images=10
                )

        return tf.nn.conv2d(x, w, stride_shape, pad) + b
