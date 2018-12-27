import tensorflow as tf
from baselines.her.util import store_args, nn


class ActorCritic:
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers,
                 **kwargs):
        """The actor-critic network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (baselines.her.Normalizer): normalizer for observations
            g_stats (baselines.her.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """
        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u']

        # Prepare inputs for actor and critic.
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)
        input_pi = tf.concat(axis=1, values=[o, g])  # for actor

        # Networks.
        with tf.variable_scope('pi'):
            # self.pi_tf = self.max_u * tf.tanh(nn(
            #     input_pi, [self.hidden] * self.layers + [self.dimu]))

            # 3-Layers FC Network
            ## FC1
            fc1 = tf.layers.dense(inputs=input_pi,
                                  units=self.hidden,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                  reuse=None,
                                  name='fc1')
            fc1 = tf.nn.relu(fc1)
            ## FC2
            fc2 = tf.layers.dense(inputs=fc1,
                                  units=self.hidden,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                  reuse=None,
                                  name='fc2')
            fc2 = tf.nn.relu(fc2)
            ## FC3
            fc3 = tf.layers.dense(inputs=fc2,
                                  units=self.dimu,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                  reuse=None,
                                  name='fc3')
            self.pi_tf_fc2 = fc2
            self.pi_tf     = fc3
            
        with tf.variable_scope('Q'):
            # for policy training
            input_Q = tf.concat(axis=1, values=[o, g, self.pi_tf / self.max_u])
            self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1])
            # for critic training
            input_Q = tf.concat(axis=1, values=[o, g, self.u_tf / self.max_u])
            self._input_Q = input_Q  # exposed for tests
            self.Q_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True)
