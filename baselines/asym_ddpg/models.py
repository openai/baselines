import tensorflow as tf
import tensorflow.contrib as tc


class Model(object):
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]


class Actor(Model):
    def __init__(self, nb_actions, name='actor', layer_norm=False):
        super(Actor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm

    def __call__(self, obs, aux, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            x = obs
            x = tc.layers.conv2d(x, 32, kernel_size=(3,3), stride=2, normalizer_fn=tc.layers.layer_norm)
            x = tc.layers.conv2d(x, 32, kernel_size=(3, 3), stride=2, normalizer_fn=tc.layers.layer_norm)
            x = tc.layers.conv2d(x, 32, kernel_size=(3, 3), stride=2, normalizer_fn=tc.layers.layer_norm)
            x = tc.layers.conv2d(x, 32, kernel_size=(3, 3), stride=2, normalizer_fn=tc.layers.layer_norm)
            x = tf.layers.flatten(x)
            x = tf.concat([x, aux], axis=-1)
            x = tf.layers.dense(x, 256)
            x = tf.nn.relu(x)
            
            x = tf.layers.dense(x, 256)
            x = tf.nn.relu(x)

            
            x = tf.layers.dense(x, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.nn.tanh(x)
        return x


class Critic(Model):
    def __init__(self, name='critic', layer_norm=False):
        super(Critic, self).__init__(name=name)
        self.layer_norm = layer_norm

    def __call__(self, state, goal, action, aux, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = tf.concat([state, goal, action, aux], axis=-1)

            x = tf.layers.dense(x, 256)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, 256)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, 256)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars
