import tensorflow as tf
import tensorflow.contrib as tc
from algos.ddpg import DDPG


class Actor(DDPG):
    """
    Input to the network is the state, output is the action under a
    deterministic policy.

    The output layer activation is a tanh to keep the action between
    -action_bound and action_bound
    """
    def __init__(self, nb_actions, name='actor', layer_norm=True):
        self.name = 'actor'
        self.namespace = 'layers'
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs
            x = self.dense(inputs=x, units=64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = self.activation('relu')(x)

            x = self.dense(inputs=x, units=64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = self.activation('relu')(x)

            x = self.dense(
                inputs=x,
                units=self.nb_actions,
                kernel_initializer=tf.random_uniform_initializer(
                    minval=-3e-3,
                    maxval=3e-3
                )
            )
            x = self.activation('tanh')(x)
        return x
