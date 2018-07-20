import tensorflow as tf


def dense(input_tensor, size, name, weight_init=None, bias_init=0, weight_loss_dict=None, reuse=None):
    """
    A dense Layer
    
    :param input_tensor: ([TensorFlow Tensor]) input
    :param size: (int) number of hidden neurons
    :param name: (str) layer name
    :param weight_init: (function or int or float) initialize the weight
    :param bias_init: (function or int or float) initialize the weight
    :param weight_loss_dict: (dict) store the weight loss if not None
    :param reuse: (bool) if can be reused
    :return: ([TensorFlow Tensor]) the output of the dense Layer
    """
    with tf.variable_scope(name, reuse=reuse):
        assert len(tf.get_variable_scope().name.split('/')) == 2

        weight = tf.get_variable("w", [input_tensor.get_shape()[1], size], initializer=weight_init)
        bias = tf.get_variable("b", [size], initializer=tf.constant_initializer(bias_init))
        weight_decay_fc = 3e-4

        if weight_loss_dict is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(weight), weight_decay_fc, name='weight_decay_loss')
            weight_loss_dict[weight] = weight_decay_fc
            weight_loss_dict[bias] = 0.0

            tf.add_to_collection(tf.get_variable_scope().name.split('/')[0] + '_' + 'losses', weight_decay)

        return tf.nn.bias_add(tf.matmul(input_tensor, weight), bias)


def kl_div(action_dist1, action_dist2, action_size):
    """
    Kullback leiber divergence
    
    :param action_dist1: ([TensorFlow Tensor]) action distribution 1
    :param action_dist2: ([TensorFlow Tensor]) action distribution 2
    :param action_size: (int) the shape of an action
    :return: (float) Kullback leiber divergence
    """
    mean1, std1 = action_dist1[:, :action_size], action_dist1[:, action_size:]
    mean2, std2 = action_dist2[:, :action_size], action_dist2[:, action_size:]

    numerator = tf.square(mean1 - mean2) + tf.square(std1) - tf.square(std2)
    denominator = 2 * tf.square(std2) + 1e-8
    return tf.reduce_sum(
        numerator / denominator + tf.log(std2) - tf.log(std1), reduction_indices=-1)
