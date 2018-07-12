import tensorflow as tf

from baselines.her.util import nn


class ActorCritic:
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers, **kwargs):
        """The actor-critic network and related training code.

        :param inputs_tf: ({str: TensorFlow Tensor}) all necessary inputs for the network: the
            observation (o), the goal (g), and the action (u)
        :param dimo: (int) the dimension of the observations
        :param dimg: (int) the dimension of the goals
        :param dimu: (int) the dimension of the actions
        :param max_u: (float) the maximum magnitude of actions; action outputs will be scaled accordingly
        :param o_stats (baselines.her.Normalizer): normalizer for observations
        :param g_stats (baselines.her.Normalizer): normalizer for goals
        :param hidden (int): number of hidden units that should be used in hidden layers
        :param layers (int): number of hidden layers
        """
        self.inputs_tf = inputs_tf
        self.dimo = dimo
        self.dimg = dimg
        self.dimu = dimu
        self.max_u = max_u
        self.o_stats = o_stats
        self.g_stats = g_stats
        self.hidden = hidden
        self.layers = layers

        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u']

        # Prepare inputs for actor and critic.
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)
        input_pi = tf.concat(axis=1, values=[o, g])  # for actor

        # Networks.
        with tf.variable_scope('pi'):
            self.pi_tf = self.max_u * tf.tanh(nn(
                input_pi, [self.hidden] * self.layers + [self.dimu]))
        with tf.variable_scope('Q'):
            # for policy training
            input_Q = tf.concat(axis=1, values=[o, g, self.pi_tf / self.max_u])
            self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1])
            # for critic training
            input_Q = tf.concat(axis=1, values=[o, g, self.u_tf / self.max_u])
            self._input_Q = input_Q  # exposed for tests
            self.Q_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True)
