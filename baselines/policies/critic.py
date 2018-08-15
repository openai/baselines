import tensorflow as tf
import tensorflow.contrib as tc
from algos.ddpg import DDPG


class Critic(DDPG):
    """
    Input to the network is the state and action, output is Q(s, a).
    The action must be obtained from the output of the Actor network
    """
    def __init__(self, name='critic', layer_norm=True):
        # super(Critic, self).__init__()
        self.name = 'critic'
        self.namespace = 'layers'
        self.layer_norm = layer_norm

    def __call__(self, obs, action, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs
            x = self.dense(inputs=x, units=64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = self.activation('relu')(x)

            x = tf.concat([x, action], axis=-1)
            x = self.dense(inputs=x, units=64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = self.activation('relu')(x)

            x = self.dense(
                inputs=x,
                units=1,
                kernel_initializer=tf.random_uniform_initializer(
                    minval=-3e-3,
                    maxval=3e-3
                )
            )
        return x
