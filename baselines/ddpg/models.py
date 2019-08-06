import tensorflow as tf
from baselines.common.models import get_network_builder


class Model(tf.keras.Model):
    def __init__(self, name, network='mlp', **network_kwargs):
        super(Model, self).__init__(name=name)
        self.network = network
        self.network_kwargs = network_kwargs

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_variables if 'layer_normalization' not in var.name]


class Actor(Model):
    def __init__(self, nb_actions, ob_shape, name='actor', network='mlp', **network_kwargs):
        super().__init__(name=name, network=network, **network_kwargs)
        self.nb_actions = nb_actions
        self.network_builder = get_network_builder(network)(**network_kwargs)(ob_shape)
        self.output_layer = tf.keras.layers.Dense(units=self.nb_actions,
                                                  activation=tf.keras.activations.tanh,
                                                  kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        output = self.output_layer(self.network_builder.outputs[0])

    @tf.function
    def call(self, obs):
        return self.output_layer(self.network_builder(obs))


class Critic(Model):
    def __init__(self, nb_actions, ob_shape, name='critic', network='mlp', **network_kwargs):
        super().__init__(name=name, network=network, **network_kwargs)
        self.layer_norm = True
        self.network_builder = get_network_builder(network)(**network_kwargs)((ob_shape[0] + nb_actions,))
        self.output_layer = tf.keras.layers.Dense(units=1,
                                                  kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),
                                                  name='output')
        output = self.output_layer(self.network_builder.outputs[0])

    @tf.function
    def call(self, obs, actions):
        x = tf.concat([obs, actions], axis=-1) # this assumes observation and action can be concatenated
        x = self.network_builder(x)
        return self.output_layer(x)

    @property
    def output_vars(self):
        return self.output_layer.trainable_variables
