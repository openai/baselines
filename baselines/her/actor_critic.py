import tensorflow as tf
from baselines.her.util import store_args, nn


def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / (stats.std + 1e-8)


def denormalize(x, stats):
    if stats is None:
        return x
    return x * stats.std + stats.mean

class ActorCritic:
    @store_args
    def __init__(self, name, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers,
                 **kwargs):
        """The actor-critic network and related training code.

        Args:
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
        # self.o_tf = inputs_tf['o']
        # self.g_tf = inputs_tf['g']
        # self.u_tf = inputs_tf['u']

        # Prepare inputs for actor and critic.
        # o = self.o_stats.normalize(self.o_tf)
        # g = self.g_stats.normalize(self.g_tf)
        # input_pi = tf.concat(axis=1, values=[o, g])  # for actor

        input_pi_shape = dimo + dimg 
        self.actor_network = nn(
            input_shape=input_pi_shape,
            layers_sizes=[self.hidden] + self.layers + [self.dimu],
            name='pi',
            output_activation='tanh')
        input_Q_shape = dimo + dimg + dimu
        self.critic_network = nn(
            input_shape=input_Q_shape,
            layers_sizes=[self.hidden] + self.layers + [1],
            name='Q')

        # Networks.
        # with tf.variable_scope('pi'):
        #     self.pi_tf = self.max_u * tf.tanh(nn(
        #         input_pi, [self.hidden] * self.layers + [self.dimu]))
        # with tf.variable_scope('Q'):
        #     # for policy training
        #     input_Q = tf.concat(axis=1, values=[o, g, self.pi_tf / self.max_u])
        #     self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1])
        #     # for critic training
        #     input_Q = tf.concat(axis=1, values=[o, g, self.u_tf / self.max_u])
        #     self._input_Q = input_Q  # exposed for tests
        #     self.Q_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True)
