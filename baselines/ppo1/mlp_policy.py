import tensorflow as tf
import gym

from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as tf_util
from baselines.common.distributions import make_proba_dist_type

class BasePolicy(object):
    def __init__(self):
        super(BasePolicy, self).__init__()
        self.sess = None

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


class MlpPolicy(BasePolicy):
    recurrent = False

    def __init__(self, name, *args, sess=None, reuse=False, **kwargs):
        super(BasePolicy, self).__init__()
        self.reuse = reuse
        self.name = name
        self._init(*args, **kwargs)
        self.scope = tf.get_variable_scope().name
        self.sess = sess

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_proba_dist_type(ac_space)
        sequence_length = None

        ob = tf_util.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        with tf.variable_scope(self.name + "/obfilter", reuse=self.reuse):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        with tf.variable_scope(self.name + '/vf', reuse=self.reuse):
            obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            last_out = obz
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name="fc%i" % (i+1),
                                                      kernel_initializer=tf_util.normc_initializer(1.0)))
            self.vpred = tf.layers.dense(last_out, 1, name='final',
                                         kernel_initializer=tf_util.normc_initializer(1.0))[:, 0]

        with tf.variable_scope(self.name + '/pol', reuse=self.reuse):
            last_out = obz
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name='fc%i' % (i+1),
                                                      kernel_initializer=tf_util.normc_initializer(1.0)))
            if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                mean = tf.layers.dense(last_out, pdtype.param_shape()[0] // 2, name='final',
                                       kernel_initializer=tf_util.normc_initializer(0.01))
                logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2],
                                         initializer=tf.zeros_initializer())
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            else:
                pdparam = tf.layers.dense(last_out, pdtype.param_shape()[0], name='final',
                                          kernel_initializer=tf_util.normc_initializer(0.01))

        self.pd = pdtype.probability_distribution_from_flat(pdparam)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = tf_util.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = tf_util.function([stochastic, ob], [ac, self.vpred])
