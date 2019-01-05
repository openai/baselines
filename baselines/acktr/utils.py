import tensorflow as tf

def dense(x, size, name, weight_init=None, bias_init=0, weight_loss_dict=None, reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        assert (len(tf.get_variable_scope().name.split('/')) == 2)

        w = tf.get_variable("w", [x.get_shape()[1], size], initializer=weight_init)
        b = tf.get_variable("b", [size], initializer=tf.constant_initializer(bias_init))
        weight_decay_fc = 3e-4

        if weight_loss_dict is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(w), weight_decay_fc, name='weight_decay_loss')
            if weight_loss_dict is not None:
                weight_loss_dict[w] = weight_decay_fc
                weight_loss_dict[b] = 0.0

            tf.add_to_collection(tf.get_variable_scope().name.split('/')[0] + '_' + 'losses', weight_decay)

        return tf.nn.bias_add(tf.matmul(x, w), b)

def kl_div(action_dist1, action_dist2, action_size):
    mean1, std1 = action_dist1[:, :action_size], action_dist1[:, action_size:]
    mean2, std2 = action_dist2[:, :action_size], action_dist2[:, action_size:]

    numerator = tf.square(mean1 - mean2) + tf.square(std1) - tf.square(std2)
    denominator = 2 * tf.square(std2) + 1e-8
    return tf.reduce_sum(
        numerator/denominator + tf.log(std2) - tf.log(std1),reduction_indices=-1)
