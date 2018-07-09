import os
from collections import deque

import numpy as np
import tensorflow as tf


def sample(logits):
    """
    Creates a sampling Tensor for non deterministic policies
    :param logits: (TensorFlow Tensor) The input probability for each action
    :return: (TensorFlow Tensor) The sampled action
    """
    noise = tf.random_uniform(tf.shape(logits))
    return tf.argmax(logits - tf.log(-tf.log(noise)), 1)


def calc_entropy(logits):
    """
    Calculates the entropy of the output values of the network
    :param logits: (TensorFlow Tensor) The input probability for each action
    :return: (TensorFlow Tensor) The Entropy of the output values of the network
    """
    a0 = logits - tf.reduce_max(logits, 1, keep_dims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, 1, keep_dims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)


def calc_entropy_softmax(p0):
    """
    Calculates the softmax entropy of the output values of the network
    :param p0: (TensorFlow Tensor) The input probability for each action
    :return: (TensorFlow Tensor) The softmax entropy of the output values of the network
    """
    return - tf.reduce_sum(p0 * tf.log(p0 + 1e-6), axis=1)


def mse(pred, target):
    """
    Returns the Mean squared error between prediction and target
    :param pred: (TensorFlow Tensor) The predicted value
    :param target: (TensorFlow Tensor) The target value
    :return: (TensorFlow Tensor) The Mean squared error between prediction and target
    """
    return tf.reduce_mean(tf.square(pred - target))


def ortho_init(scale=1.0):
    """
    Orthogonal initialization for the policy weights
    :param scale: (float) The output scale
    :return: (function) an initialization function for the weights
    """
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


def conv(x, scope, *, nf, rf, stride, pad='VALID', init_scale=1.0, data_format='NHWC', one_dim_bias=False):
    """
    Creates a 2d convolutional layer for TensorFlow
    :param x: (TensorFlow Tensor) The input tensor for the convolution
    :param scope: (str) The TensorFlow variable scope
    :param nf: (int) The number of filters
    :param rf: (int) The filter size
    :param stride: (int) The stride of the convolution
    :param pad: (str) The padding type ('VALID' or 'SAME')
    :param init_scale: (int) The initialization scale
    :param data_format: (str) The data format for the convolution weights
    :param one_dim_bias: (bool) If the bias should be one dimentional or not
    :return: (TensorFlow Tensor) 2d convolutional layer
    """
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
        w = tf.get_variable("w", wshape, initializer=ortho_init(init_scale))
        b = tf.get_variable("b", bias_var_shape, initializer=tf.constant_initializer(0.0))
        if not one_dim_bias and data_format == 'NHWC':
            b = tf.reshape(b, bshape)
        return b + tf.nn.conv2d(x, w, strides=strides, padding=pad, data_format=data_format)


def fc(x, scope, nh, *, init_scale=1.0, init_bias=0.0):
    """
    Creates a fully connected layer for TensorFlow
    :param x: (TensorFlow Tensor) The input tensor for the fully connected layer
    :param scope: (str) The TensorFlow variable scope
    :param nh: (int) The number of hidden neurons
    :param init_scale: (int) The initialization scale
    :param init_bias: (int) The initialization offset bias
    :return: (TensorFlow Tensor) fully connected layer
    """
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias))
        return tf.matmul(x, w) + b


def batch_to_seq(h, nbatch, nsteps, flat=False):
    """
    Transform a batch of Tensors, into a sequence of Tensors for reccurent policies
    :param h: (TensorFlow Tensor) The input tensor to unroll
    :param nbatch: (int) The number of batch to run (nenvs * nsteps)
    :param nsteps: (int) The number of steps to run for each environment
    :param flat: (bool) If the input Tensor is flat
    :return: (TensorFlow Tensor) sequence of Tensors for reccurent policies
    """
    if flat:
        h = tf.reshape(h, [nbatch, nsteps])
    else:
        h = tf.reshape(h, [nbatch, nsteps, -1])
    return [tf.squeeze(v, [1]) for v in tf.split(axis=1, num_or_size_splits=nsteps, value=h)]


def seq_to_batch(h, flat=False):
    """
    Transform a sequence of Tensors, into a batch of Tensors for reccurent policies
    :param h: (TensorFlow Tensor) The input tensor to batch
    :param flat: (bool) If the input Tensor is flat
    :return: (TensorFlow Tensor) batch of Tensors for reccurent policies
    """
    shape = h[0].get_shape().as_list()
    if not flat:
        assert (len(shape) > 1)
        nh = h[0].get_shape()[-1].value
        return tf.reshape(tf.concat(axis=1, values=h), [-1, nh])
    else:
        return tf.reshape(tf.stack(values=h, axis=1), [-1])


def lstm(xs, ms, s, scope, nh, init_scale=1.0):
    """
    Creates an Long Short Term Memory (LSTM) cell for TensorFlow
    :param xs: (TensorFlow Tensor) The input tensor for the LSTM cell
    :param ms: (TensorFlow Tensor) The mask tensor for the LSTM cell
    :param s: (TensorFlow Tensor) The state tensor for the LSTM cell
    :param scope: (str) The TensorFlow variable scope
    :param nh: (int) The number of hidden neurons
    :param init_scale: (int) The initialization scale
    :return: (TensorFlow Tensor) LSTM cell
    """
    nbatch, nin = [v.value for v in xs[0].get_shape()]
    with tf.variable_scope(scope):
        wx = tf.get_variable("wx", [nin, nh * 4], initializer=ortho_init(init_scale))
        wh = tf.get_variable("wh", [nh, nh * 4], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [nh * 4], initializer=tf.constant_initializer(0.0))

    c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
    for idx, (x, m) in enumerate(zip(xs, ms)):
        c = c * (1 - m)
        h = h * (1 - m)
        z = tf.matmul(x, wx) + tf.matmul(h, wh) + b
        i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        u = tf.tanh(u)
        c = f * c + i * u
        h = o * tf.tanh(c)
        xs[idx] = h
    s = tf.concat(axis=1, values=[c, h])
    return xs, s


def _ln(x, g, b, e=1e-5, axes=None):
    """
    Creates a normalizing layer for TensorFlow
    :param x: (TensorFlow Tensor) The input tensor for the Layer normalization
    :param g: (TensorFlow Tensor) The scale tensor for the Layer normalization
    :param b: (TensorFlow Tensor) The bias tensor for the Layer normalization
    :param e: (float) The epsilon value for floating point calculations
    :param axes: (tuple, list or int) The axes to apply the mean and variance calculation
    :return: (TensorFlow Tensor) a normalizing layer
    """
    if axes is None:
        axes = [1]
    u, s = tf.nn.moments(x, axes=axes, keep_dims=True)
    x = (x - u) / tf.sqrt(s + e)
    x = x * g + b
    return x


def lnlstm(xs, ms, s, scope, nh, init_scale=1.0):
    """
    Creates a Layer normalized LSTM (LNLSTM) cell for TensorFlow
    :param xs: (TensorFlow Tensor) The input tensor for the LSTM cell
    :param ms: (TensorFlow Tensor) The mask tensor for the LSTM cell
    :param s: (TensorFlow Tensor) The state tensor for the LSTM cell
    :param scope: (str) The TensorFlow variable scope
    :param nh: (int) The number of hidden neurons
    :param init_scale: (int) The initialization scale
    :return: (TensorFlow Tensor) LNLSTM cell
    """
    nbatch, nin = [v.value for v in xs[0].get_shape()]
    with tf.variable_scope(scope):
        wx = tf.get_variable("wx", [nin, nh * 4], initializer=ortho_init(init_scale))
        gx = tf.get_variable("gx", [nh * 4], initializer=tf.constant_initializer(1.0))
        bx = tf.get_variable("bx", [nh * 4], initializer=tf.constant_initializer(0.0))

        wh = tf.get_variable("wh", [nh, nh * 4], initializer=ortho_init(init_scale))
        gh = tf.get_variable("gh", [nh * 4], initializer=tf.constant_initializer(1.0))
        bh = tf.get_variable("bh", [nh * 4], initializer=tf.constant_initializer(0.0))

        b = tf.get_variable("b", [nh * 4], initializer=tf.constant_initializer(0.0))

        gc = tf.get_variable("gc", [nh], initializer=tf.constant_initializer(1.0))
        bc = tf.get_variable("bc", [nh], initializer=tf.constant_initializer(0.0))

    c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
    for idx, (x, m) in enumerate(zip(xs, ms)):
        c = c * (1 - m)
        h = h * (1 - m)
        z = _ln(tf.matmul(x, wx), gx, bx) + _ln(tf.matmul(h, wh), gh, bh) + b
        i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        u = tf.tanh(u)
        c = f * c + i * u
        h = o * tf.tanh(_ln(c, gc, bc))
        xs[idx] = h
    s = tf.concat(axis=1, values=[c, h])
    return xs, s


def conv_to_fc(x):
    """
    Reshapes a Tensor from a convolutional network to a Tensor for a fully connected network
    :param x: (TensorFlow Tensor) The convolutional input tensor
    :return: (TensorFlow Tensor) The fully connected output tensor
    """
    nh = np.prod([v.value for v in x.get_shape()[1:]])
    x = tf.reshape(x, [-1, nh])
    return x


def discount_with_dones(rewards, dones, gamma):
    """
    Apply the discount value to the reward, where the environment is not done
    :param rewards: ([float]) The rewards
    :param dones: ([bool]) Whether an environment is done or not
    :param gamma: (float) The discount value
    :return: ([float]) The discounted rewards
    """
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r * (1. - done)  # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]


def find_trainable_variables(key):
    """
    Returns the trainable variables within a given scope
    :param key: (str) The variable scope
    :return: ([TensorFlow Tensor]) the trainable variables
    """
    with tf.variable_scope(key):
        return tf.trainable_variables()


def make_path(f):
    """
    For a given path, create the folders if they do not exist
    :param f: (str) The path
    :return: (bool) Whether or not it finished correctly
    """
    return os.makedirs(f, exist_ok=True)


def constant(_):
    """
    Returns a constant value for the Scheduler
    :param _: ignored
    :return: (float) 1
    """
    return 1.


def linear(p):
    """
    Returns a linear value for the Scheduler
    :param p: (float) The input value
    :return: (float) 1 - p
    """
    return 1 - p


def middle_drop(p):
    """
    Returns a linear value with a drop near the middle to a constant value for the Scheduler
    :param p: (float) The input value
    :return: (float) 1 - p if (1 - p) >= 0.75 else 0.075
    """
    eps = 0.75
    if 1 - p < eps:
        return eps * 0.1
    return 1 - p


def double_linear_con(p):
    """
    Returns a linear value (x2) with a flattend tail for the Scheduler
    :param p: (float) The input value
    :return: (float) 1 - p*2 if (1 - p*2) >= 0.125 else 0.125
    """
    p *= 2
    eps = 0.125
    if 1 - p < eps:
        return eps
    return 1 - p


def double_middle_drop(p):
    """
    Returns a linear value with two drops near the middle to a constant value for the Scheduler
    :param p: (float) The input value
    :return: (float) if 0.75 <= 1 - p: 1 - p, if 0.25 <= 1 - p < 0.75: 0.75, if 1 - p < 0.25: 0.125
    """
    eps1 = 0.75
    eps2 = 0.25
    if 1 - p < eps1:
        if 1 - p < eps2:
            return eps2 * 0.5
        return eps1 * 0.1
    return 1 - p


schedules = {
    'linear': linear,
    'constant': constant,
    'double_linear_con': double_linear_con,
    'middle_drop': middle_drop,
    'double_middle_drop': double_middle_drop
}


class Scheduler(object):
    def __init__(self, v, nvalues, schedule):
        """
        Update a value every iteration, with a specific curve
        :param v: (float) initial value
        :param nvalues: (int) the total number of iterations
        :param schedule: (function) the curve you wish to follow for your value
        """
        self.n = 0.
        self.v = v
        self.nvalues = nvalues
        self.schedule = schedules[schedule]

    def value(self):
        """
        Update the Scheduler, and return the current value
        :return: (float) the current value
        """
        current_value = self.v * self.schedule(self.n / self.nvalues)
        self.n += 1.
        return current_value

    def value_steps(self, steps):
        """
        Get a value for a given step
        :param steps: (int) The current number of iterations
        :return: (float) the value for the current number of iterations
        """
        return self.v * self.schedule(steps / self.nvalues)


class EpisodeStats:
    def __init__(self, nsteps, nenvs):
        """
        Calculates the episode statistics
        :param nsteps: (int) The number of steps to run for each environment
        :param nenvs: (int) The number of environments
        """
        self.episode_rewards = []
        for i in range(nenvs):
            self.episode_rewards.append([])
        self.lenbuffer = deque(maxlen=40)  # rolling buffer for episode lengths
        self.rewbuffer = deque(maxlen=40)  # rolling buffer for episode rewards
        self.nsteps = nsteps
        self.nenvs = nenvs

    def feed(self, rewards, masks):
        """
        Update the latest reward and mask
        :param rewards: ([float]) The new rewards for the new step
        :param masks: ([float]) The new masks for the new step
        """
        rewards = np.reshape(rewards, [self.nenvs, self.nsteps])
        masks = np.reshape(masks, [self.nenvs, self.nsteps])
        for i in range(0, self.nenvs):
            for j in range(0, self.nsteps):
                self.episode_rewards[i].append(rewards[i][j])
                if masks[i][j]:
                    reward_length = len(self.episode_rewards[i])
                    reward_sum = sum(self.episode_rewards[i])
                    self.lenbuffer.append(reward_length)
                    self.rewbuffer.append(reward_sum)
                    self.episode_rewards[i] = []

    def mean_length(self):
        """
        Returns the average length of each episode
        :return: (float)
        """
        if self.lenbuffer:
            return np.mean(self.lenbuffer)
        else:
            return 0  # on the first params dump, no episodes are finished

    def mean_reward(self):
        """
        Returns the average reward of each episode
        :return: (float)
        """
        if self.rewbuffer:
            return np.mean(self.rewbuffer)
        else:
            return 0


# For ACER
def get_by_index(x, idx):
    """
    Return the input tensor, offset by a certain value
    :param x: (TensorFlow Tensor) The input tensor
    :param idx: (int) The index offset
    :return: (TensorFlow Tensor) the offset tensor
    """
    assert (len(x.get_shape()) == 2)
    assert (len(idx.get_shape()) == 1)
    idx_flattened = tf.range(0, x.shape[0]) * x.shape[1] + idx
    y = tf.gather(tf.reshape(x, [-1]),  # flatten input
                  idx_flattened)  # use flattened indices
    return y


def check_shape(ts, shapes):
    """
    Verifies the tensors match the given shape, will raise an error if the shapes do not match
    :param ts: ([TensorFlow Tensor]) The tensors that should be checked
    :param shapes: ([list]) The list of shapes for each tensor
    """
    i = 0
    for (t, shape) in zip(ts, shapes):
        assert t.get_shape().as_list() == shape, "id " + str(i) + " shape " + str(t.get_shape()) + str(shape)
        i += 1


def avg_norm(t):
    """
    Return an average of the L2 normalization of the batch
    :param t: (TensorFlow Tensor) The input tensor
    :return: (TensorFlow Tensor) Average L2 normalization of the batch
    """
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(t), axis=-1)))


def gradient_add(g1, g2, param):
    """
    Sum two gradiants
    :param g1: (TensorFlow Tensor) The first gradiant
    :param g2: (TensorFlow Tensor) The second gradiant
    :param param: (TensorFlow parameters) The trainable parameters
    :return: (TensorFlow Tensor) the sum of the gradiants
    """
    print([g1, g2, param.name])
    assert (not (g1 is None and g2 is None)), param.name
    if g1 is None:
        return g2
    elif g2 is None:
        return g1
    else:
        return g1 + g2


def q_explained_variance(qpred, q):
    """
    Calculates the explained variance of the Q value
    :param qpred: (TensorFlow Tensor) The predicted Q value
    :param q: (TensorFlow Tensor) The expected Q value
    :return: (TensorFlow Tensor) the explained variance of the Q value
    """
    _, vary = tf.nn.moments(q, axes=[0, 1])
    _, varpred = tf.nn.moments(q - qpred, axes=[0, 1])
    check_shape([vary, varpred], [[]] * 2)
    return 1.0 - (varpred / vary)
