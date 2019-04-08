import numpy as np
import tensorflow as tf

from baselines.a2c.utils import ortho_init, fc, lstm, lnlstm
from baselines.common.models import register, nature_cnn, RNN


@register("ppo_lstm", is_rnn=True)
def ppo_lstm(num_units=128, layer_norm=False):
    def _network_fn(input, mask, state):
        input = tf.layers.flatten(input)
        mask = tf.to_float(mask)

        if layer_norm:
            h, next_state = lnlstm([input], [mask[:, None]], state, scope='lnlstm', nh=num_units)
        else:
            h, next_state = lstm([input], [mask[:, None]], state, scope='lstm', nh=num_units)
        h = h[0]
        return h, next_state

    return RNN(_network_fn, memory_size=num_units * 2)


@register("ppo_cnn_lstm", is_rnn=True)
def ppo_cnn_lstm(num_units=128, layer_norm=False, **conv_kwargs):
    def _network_fn(input, mask, state):
        mask = tf.to_float(mask)
        initializer = ortho_init(np.sqrt(2))

        h = nature_cnn(input, **conv_kwargs)
        h = tf.layers.flatten(h)
        h = tf.layers.dense(h, units=512, activation=tf.nn.relu, kernel_initializer=initializer)

        if layer_norm:
            h, next_state = lnlstm([h], [mask[:, None]], state, scope='lnlstm', nh=num_units)
        else:
            h, next_state = lstm([h], [mask[:, None]], state, scope='lstm', nh=num_units)
        h = h[0]
        return h, next_state

    return RNN(_network_fn, memory_size=num_units * 2)


@register("ppo_cnn_lnlstm", is_rnn=True)
def ppo_cnn_lnlstm(num_units=128, **conv_kwargs):
    return ppo_cnn_lstm(num_units, layer_norm=True, **conv_kwargs)


@register("ppo_lstm_mlp", is_rnn=True)
def ppo_lstm_mlp(num_units=128, layer_norm=False):
    def network_fn(input, mask):
        memory_size = num_units * 2
        nbatch = input.shape[0]
        mask.get_shape().assert_is_compatible_with([nbatch])
        state = tf.Variable(np.zeros([nbatch, memory_size]),
                            name='lstm_state',
                            trainable=False,
                            dtype=tf.float32,
                            collections=[tf.GraphKeys.LOCAL_VARIABLES])

        def _network_fn(input, mask, state):
            input = tf.layers.flatten(input)
            mask = tf.to_float(mask)

            h, next_state = lstm([input], [mask[:, None]], state, scope='lstm', nh=num_units)
            h = h[0]

            num_layers = 2
            num_hidden = 64
            activation = tf.nn.relu
            for i in range(num_layers):
                h = fc(h, 'mlp_fc{}'.format(i), nh=num_hidden, init_scale=np.sqrt(2))
                h = activation(h)
            return h, next_state

        return state, RNN(_network_fn)

    return RNN(network_fn)
