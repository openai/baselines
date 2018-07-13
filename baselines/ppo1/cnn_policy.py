import tensorflow as tf

import baselines.common.tf_util as tf_util
from baselines.ppo1.mlp_policy import BasePolicy


class CnnPolicy(BasePolicy):
    recurrent = False

    def __init__(self, name, ob_space, ac_space, architecture_size='large', sess=None, reuse=False):
        """

        :param name: (str) type of the policy (lin, logits, value)
        :param ob_space: (Gym Space) The observation space of the environment
        :param ac_space: (Gym Space) The action space of the environment
        :param architecture_size: (str) size of the policy's architecture
               (small as in A3C paper, large as in Nature DQN)
        :param sess: (TensorFlow session) The current TensorFlow session containing the variables.
        :param reuse: (bool) If the policy is reusable or not
        """
        super(CnnPolicy, self).__init__()
        self.reuse = reuse
        self.name = name
        self._init(ob_space, ac_space, architecture_size)
        self.scope = tf.get_variable_scope().name
        self.sess = sess

    def _init(self, ob_space, ac_space, architecture_size):
        """

        :param ob_space: (Gym Space) The observation space of the environment
        :param ac_space: (Gym Space) The action space of the environment
        :param architecture_size: (str) size of the policy's architecture
               (small as in A3C paper, large as in Nature DQN)
        """
        ob, pdtype = self.get_obs_and_pdtype(ob_space, ac_space)

        with tf.variable_scope(self.name, reuse=self.reuse):
            normalized_obs = ob / 255.0
            if architecture_size == 'small':  # from A3C paper
                layer_1 = tf.nn.relu(tf_util.conv2d(normalized_obs, 16, "l1", [8, 8], [4, 4], pad="VALID"))
                layer_2 = tf.nn.relu(tf_util.conv2d(layer_1, 32, "l2", [4, 4], [2, 2], pad="VALID"))
                flattened_layer_2 = tf_util.flattenallbut0(layer_2)
                last_layer = tf.nn.relu(tf.layers.dense(flattened_layer_2, 256,
                                                        name='lin', kernel_initializer=tf_util.normc_initializer(1.0)))
            elif architecture_size == 'large':  # Nature DQN
                layer_1 = tf.nn.relu(tf_util.conv2d(normalized_obs, 32, "l1", [8, 8], [4, 4], pad="VALID"))
                layer_2 = tf.nn.relu(tf_util.conv2d(layer_1, 64, "l2", [4, 4], [2, 2], pad="VALID"))
                layer_3 = tf.nn.relu(tf_util.conv2d(layer_2, 64, "l3", [3, 3], [1, 1], pad="VALID"))
                flattened_layer_3 = tf_util.flattenallbut0(layer_3)
                last_layer = tf.nn.relu(tf.layers.dense(flattened_layer_3, 512,
                                                        name='lin', kernel_initializer=tf_util.normc_initializer(1.0)))
            else:
                raise NotImplementedError

            logits = tf.layers.dense(last_layer, pdtype.param_shape()[0], name='logits',
                                     kernel_initializer=tf_util.normc_initializer(0.01))

            self.pd = pdtype.proba_distribution_from_flat(logits)
            self.vpred = tf.layers.dense(last_layer, 1, name='value', kernel_initializer=tf_util.normc_initializer(1.0))[:, 0]

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = self.pd.sample()
        self._act = tf_util.function([stochastic, ob], [ac, self.vpred])
