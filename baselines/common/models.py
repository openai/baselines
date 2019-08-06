import numpy as np
import tensorflow as tf
from baselines.a2c import utils
from baselines.a2c.utils import ortho_init, conv

mapping = {}

def register(name):
    def _thunk(func):
        mapping[name] = func
        return func
    return _thunk


def nature_cnn(input_shape, **conv_kwargs):
    """
    CNN from Nature paper.
    """
    print('input shape is {}'.format(input_shape))
    x_input = tf.keras.Input(shape=input_shape, dtype=tf.uint8)
    h = x_input
    h = tf.cast(h, tf.float32) / 255.
    h = conv('c1', nf=32, rf=8, stride=4, activation='relu', init_scale=np.sqrt(2))(h)
    h2 = conv('c2', nf=64, rf=4, stride=2, activation='relu', init_scale=np.sqrt(2))(h)
    h3 = conv('c3', nf=64, rf=3, stride=1, activation='relu', init_scale=np.sqrt(2))(h2)
    h3 = tf.keras.layers.Flatten()(h3)
    h3 = tf.keras.layers.Dense(units=512, kernel_initializer=ortho_init(np.sqrt(2)),
                               name='fc1', activation='relu')(h3)
    network = tf.keras.Model(inputs=[x_input], outputs=[h3])
    return network

@register("mlp")
def mlp(num_layers=2, num_hidden=64, activation=tf.tanh):
    """
    Stack of fully-connected layers to be used in a policy / q-function approximator

    Parameters:
    ----------

    num_layers: int                 number of fully-connected layers (default: 2)

    num_hidden: int                 size of fully-connected layers (default: 64)

    activation:                     activation function (default: tf.tanh)

    Returns:
    -------

    function that builds fully connected network with a given input tensor / placeholder
    """
    def network_fn(input_shape):
        print('input shape is {}'.format(input_shape))
        x_input = tf.keras.Input(shape=input_shape)
        # h = tf.keras.layers.Flatten(x_input)
        h = x_input
        for i in range(num_layers):
          h = tf.keras.layers.Dense(units=num_hidden, kernel_initializer=ortho_init(np.sqrt(2)),
                                    name='mlp_fc{}'.format(i), activation=activation)(h)

        network = tf.keras.Model(inputs=[x_input], outputs=[h])
        return network

    return network_fn


@register("cnn")
def cnn(**conv_kwargs):
    def network_fn(input_shape):
        return nature_cnn(input_shape, **conv_kwargs)
    return network_fn


@register("conv_only")
def conv_only(convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], **conv_kwargs):
    '''
    convolutions-only net
    Parameters:
    ----------
    conv:       list of triples (filter_number, filter_size, stride) specifying parameters for each layer.
    Returns:
    function that takes tensorflow tensor as input and returns the output of the last convolutional layer
    '''

    def network_fn(input_shape):
        print('input shape is {}'.format(input_shape))
        x_input = tf.keras.Input(shape=input_shape, dtype=tf.uint8)
        h = x_input
        h = tf.cast(h, tf.float32) / 255.
        with tf.name_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                h = tf.keras.layers.Conv2D(
                    filters=num_outputs, kernel_size=kernel_size, strides=stride,
                    activation='relu', **conv_kwargs)(h)

        network = tf.keras.Model(inputs=[x_input], outputs=[h])
        return network
    return network_fn


def get_network_builder(name):
    """
    If you want to register your own network outside models.py, you just need:

    Usage Example:
    -------------
    from baselines.common.models import register
    @register("your_network_name")
    def your_network_define(**net_kwargs):
        ...
        return network_fn

    """
    if callable(name):
        return name
    elif name in mapping:
        return mapping[name]
    else:
        raise ValueError('Unknown network type: {}'.format(name))
