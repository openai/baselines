import tensorflow as tf

import baselines.common.tf_util as tf_utils
from baselines.ppo1.mlp_policy import BasePolicy


class CnnPolicy(BasePolicy):
    recurrent = False

    def __init__(self, name, ob_space, ac_space, sess=None, reuse=False):
        """
        A CNN policy object for TRPO

        :param name: (str) type of the policy (lin, logits, value)
        :param ob_space: (Gym Space) The observation space of the environment
        :param ac_space: (Gym Space) The action space of the environment
        :param sess: (TensorFlow session) The current TensorFlow session containing the variables.
        :param reuse: (bool) If the policy is reusable or not
        """
        super(CnnPolicy, self).__init__()
        self.sess = sess
        self.reuse = reuse
        self.name = name
        self._init(ob_space, ac_space)
        self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space):
        """

        :param ob_space: (Gym Space) The observation space of the environment
        :param ac_space: (Gym Space) The action space of the environment
        """
        obs, pdtype = self.get_obs_and_pdtype(ob_space, ac_space)

        obs_normalized = obs / 255.0

        with tf.variable_scope(self.name + "/pol", reuse=self.reuse):
            layer_1 = tf.nn.relu(tf_utils.conv2d(obs_normalized, 8, "l1", [8, 8], [4, 4], pad="VALID"))
            layer_2 = tf.nn.relu(tf_utils.conv2d(layer_1, 16, "l2", [4, 4], [2, 2], pad="VALID"))
            layer_2 = tf_utils.flattenallbut0(layer_2)
            layer_3 = tf.nn.relu(tf.layers.dense(layer_2, 128, name='lin',
                                                 kernel_initializer=tf_utils.normc_initializer(1.0)))
            logits = tf.layers.dense(layer_3, pdtype.param_shape()[0], name='logits',
                                     kernel_initializer=tf_utils.normc_initializer(0.01))
            self.proba_distribution = pdtype.proba_distribution_from_flat(logits)
        with tf.variable_scope(self.name + "/vf", reuse=self.reuse):
            layer_1 = tf.nn.relu(tf_utils.conv2d(obs_normalized, 8, "l1", [8, 8], [4, 4], pad="VALID"))
            layer_2 = tf.nn.relu(tf_utils.conv2d(layer_1, 16, "l2", [4, 4], [2, 2], pad="VALID"))
            layer_2 = tf_utils.flattenallbut0(layer_2)
            layer_3 = tf.nn.relu(tf.layers.dense(layer_2, 128, name='lin',
                                                 kernel_initializer=tf_utils.normc_initializer(1.0)))
            self.vpred = tf.layers.dense(layer_3, 1, name='value',
                                         kernel_initializer=tf_utils.normc_initializer(1.0))
            self.vpredz = self.vpred

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        action = self.proba_distribution.sample()
        self._act = tf_utils.function([stochastic, obs], [action, self.vpred])
