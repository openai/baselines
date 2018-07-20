import tensorflow as tf
import tensorflow.contrib as tc


class Model(object):
    def __init__(self, name, layer_norm=True):
        """
        A TensorFlow Model type

        :param name: (str) the name of the model
        """
        self.name = name
        self.layer_norm = layer_norm

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]

    def fc_with_relu(self, input_tensor):
        """
        Fully connected layer followed by ReLU
        with optional batchnorm

        :param input_tensor: (TensorFlow Tensor) the input tensor
        :return: (TensorFlow Tensor) the output tensor
        """
        preactivation = tf.layers.dense(input_tensor, 64)
        if self.layer_norm:
            preactivation = tc.layers.layer_norm(preactivation, center=True, scale=True)
        return tf.nn.relu(preactivation)

    @classmethod
    def nature_cnn(cls, input_tensor):
        """
        CNN from Nature paper.

        :param input_tensor: (TensorFlow Tensor) the input tensor
        :return: (TensorFlow Tensor) nature CNN output tensor
        """
        layer_1 = tf.layers.conv2d(input_tensor, 32, (8, 8), (4, 4))
        layer_1 = tf.nn.relu(layer_1)

        layer_2 = tf.layers.conv2d(layer_1, 64, (4, 4), (2, 2))
        layer_2 = tf.nn.relu(layer_2)

        layer_3 = tf.layers.conv2d(layer_2, 64, (3, 3))
        layer_3 = tf.nn.relu(layer_3)

        return tc.layers.flatten(layer_3)


class ActorCNN(Model):
    def __init__(self, nb_actions, name='ActorCNN', layer_norm=True):
        """
        A TensorFlow CNN Actor model, this is used to output the actions

        :param nb_actions: (int) the size of the action space
        :param name: (str) the name of the model (default: 'actor')
        :param layer_norm: (bool) enable layer normalization
        """
        super(ActorCNN, self).__init__(name=name, layer_norm=layer_norm)
        self.nb_actions = nb_actions

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            preprocessed = self.nature_cnn(obs)

            layer_1 = self.fc_with_relu(preprocessed)
            layer_2 = self.fc_with_relu(layer_1)
            last_layer = tf.layers.dense(layer_2, self.nb_actions,
                                         kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            squashed_out = tf.nn.tanh(last_layer)
        return squashed_out


class ActorMLP(Model):
    def __init__(self, nb_actions, name='ActorMLP', layer_norm=True):
        """
        A TensorFlow MLP Actor model, this is used to output the actions

        :param nb_actions: (int) the size of the action space
        :param name: (str) the name of the model (default: 'actor')
        :param layer_norm: (bool) enable layer normalization
        """
        super(ActorMLP, self).__init__(name=name, layer_norm=layer_norm)
        self.nb_actions = nb_actions

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            layer_1 = self.fc_with_relu(obs)
            layer_2 = self.fc_with_relu(layer_1)
            last_layer = tf.layers.dense(layer_2, self.nb_actions,
                                         kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            squashed_out = tf.nn.tanh(last_layer)
        return squashed_out


class CriticCNN(Model):
    def __init__(self, name='CriticCNN', layer_norm=True):
        """
        A TensorFlow CNN Critic model, this is used to output the value of a state

        :param name: (str) the name of the model (default: 'critic')
        :param layer_norm: (bool) enable layer normalization
        """
        super(CriticCNN, self).__init__(name=name, layer_norm=layer_norm)

    def __call__(self, obs, action, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            preprocessed = self.nature_cnn(obs)

            layer_1 = self.fc_with_relu(preprocessed)
            layer_2 = tf.concat([layer_1, action], axis=-1)
            layer_3 = self.fc_with_relu(layer_2)
            value = tf.layers.dense(layer_3, 1,
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))

        return value

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars


class CriticMLP(Model):
    def __init__(self, name='CriticMLP', layer_norm=True):
        """
        A TensorFlow MLP Critic model, this is used to output the value of a state

        :param name: (str) the name of the model (default: 'critic')
        :param layer_norm: (bool) enable layer normalization
        """
        super(CriticMLP, self).__init__(name=name, layer_norm=layer_norm)

    def __call__(self, obs, action, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            layer_1 = self.fc_with_relu(obs)
            layer_2 = tf.concat([layer_1, action], axis=-1)
            layer_3 = self.fc_with_relu(layer_2)
            value = tf.layers.dense(layer_3, 1,
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return value

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars
