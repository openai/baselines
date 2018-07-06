import tensorflow as tf
import gym

import baselines.common.tf_util as tf_utils
from baselines.common.distributions import make_pdtype


class CnnPolicy(object):
    recurrent = False

    def __init__(self, name, ob_space, ac_space, sess=None, reuse=False):
        self.sess = sess
        self.reuse = reuse
        self.name = name
        self._init(ob_space, ac_space)
        self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = tf_utils.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        obscaled = ob / 255.0

        with tf.variable_scope(self.name + "/pol", reuse=self.reuse):
            x = obscaled
            x = tf.nn.relu(tf_utils.conv2d(x, 8, "l1", [8, 8], [4, 4], pad="VALID"))
            x = tf.nn.relu(tf_utils.conv2d(x, 16, "l2", [4, 4], [2, 2], pad="VALID"))
            x = tf_utils.flattenallbut0(x)
            x = tf.nn.relu(tf.layers.dense(x, 128, name='lin', kernel_initializer=tf_utils.normc_initializer(1.0)))
            logits = tf.layers.dense(x, pdtype.param_shape()[0], name='logits',
                                     kernel_initializer=tf_utils.normc_initializer(0.01))
            self.pd = pdtype.pdfromflat(logits)
        with tf.variable_scope(self.name + "/vf", reuse=self.reuse):
            x = obscaled
            x = tf.nn.relu(tf_utils.conv2d(x, 8, "l1", [8, 8], [4, 4], pad="VALID"))
            x = tf.nn.relu(tf_utils.conv2d(x, 16, "l2", [4, 4], [2, 2], pad="VALID"))
            x = tf_utils.flattenallbut0(x)
            x = tf.nn.relu(tf.layers.dense(x, 128, name='lin', kernel_initializer=tf_utils.normc_initializer(1.0)))
            self.vpred = tf.layers.dense(x, 1, name='value', kernel_initializer=tf_utils.normc_initializer(1.0))
            self.vpredz = self.vpred

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = self.pd.sample()
        self._act = tf_utils.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1 = self._act(stochastic, ob[None], sess=self.sess)
        return ac1[0], vpred1[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    @classmethod
    def get_initial_state(cls):
        return []
