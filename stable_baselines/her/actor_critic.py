import tensorflow as tf

from stable_baselines.her.util import mlp


class ActorCritic:
    def __init__(self, inputs_tf, dim_obs, dim_goal, dim_action,
                 max_u, o_stats, g_stats, hidden, layers, **kwargs):
        """The actor-critic network and related training code.

        :param inputs_tf: ({str: TensorFlow Tensor}) all necessary inputs for the network: the
            observation (o), the goal (g), and the action (u)
        :param dim_obs: (int) the dimension of the observations
        :param dim_goal: (int) the dimension of the goals
        :param dim_action: (int) the dimension of the actions
        :param max_u: (float) the maximum magnitude of actions; action outputs will be scaled accordingly
        :param o_stats (stable_baselines.her.Normalizer): normalizer for observations
        :param g_stats (stable_baselines.her.Normalizer): normalizer for goals
        :param hidden (int): number of hidden units that should be used in hidden layers
        :param layers (int): number of hidden layers
        """
        self.inputs_tf = inputs_tf
        self.dim_obs = dim_obs
        self.dim_goal = dim_goal
        self.dim_action = dim_action
        self.max_u = max_u
        self.o_stats = o_stats
        self.g_stats = g_stats
        self.hidden = hidden
        self.layers = layers

        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u']

        # Prepare inputs for actor and critic.
        obs = self.o_stats.normalize(self.o_tf)
        goals = self.g_stats.normalize(self.g_tf)
        input_pi = tf.concat(axis=1, values=[obs, goals])  # for actor

        # Networks.
        with tf.variable_scope('pi'):
            self.pi_tf = self.max_u * tf.tanh(mlp(
                input_pi, [self.hidden] * self.layers + [self.dimu]))
        with tf.variable_scope('Q'):
            # for policy training
            input_q = tf.concat(axis=1, values=[obs, goals, self.pi_tf / self.max_u])
            self.q_pi_tf = mlp(input_q, [self.hidden] * self.layers + [1])
            # for critic training
            input_q = tf.concat(axis=1, values=[obs, goals, self.u_tf / self.max_u])
            self._input_q = input_q  # exposed for tests
            self.q_tf = mlp(input_q, [self.hidden] * self.layers + [1], reuse=True)
