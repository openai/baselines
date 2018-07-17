import tensorflow as tf
import gym

from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as tf_util
from baselines.common.distributions import make_proba_dist_type


class BasePolicy(object):
    def __init__(self):
        """
        A base policy object for PPO1
        """
        super(BasePolicy, self).__init__()
        self.sess = None
        self.pdtype = None
        self._act = None
        self.scope = None
        self.obs_ph = None

    def get_obs_and_pdtype(self, ob_space, ac_space):
        """
        Initialize probability distribution and get observation placeholder.

        :param ob_space: (Gym Spaces) the observation space
        :param ac_space: (Gym Spaces) the action space
        """
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_proba_dist_type(ac_space)
        sequence_length = None

        if self.obs_ph is None:
            self.obs_ph = tf.placeholder(dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape), name="ob")

        return self.obs_ph, pdtype

    def act(self, stochastic, obs):
        """
        Get the action from the policy, using the observation

        :param stochastic: (bool) whether or not to use a stochastic or deterministic policy
        :param obs: (TensorFlow Tensor or numpy Number) the observation
        :return: (numpy Number, numpy Number) the action and value function
        """
        ac1, vpred1 = self._act(stochastic, obs[None], sess=self.sess)
        return ac1[0], vpred1[0]

    def get_variables(self):
        """
        Get all the policy's variables

        :return: ([TensorFlow Tensor]) the variables of the network
        """
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        """
        Get the policy's trainable variables

        :return: ([TensorFlow Tensor]) the trainable variables of the network
        """
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    @classmethod
    def get_initial_state(cls):
        """
        Get the initial state

        :return: ([numpy Number]) the initial state
        """
        return []


class MlpPolicy(BasePolicy):
    recurrent = False

    def __init__(self, name, *args, sess=None, reuse=False, **kwargs):
        """
        A MLP policy object for PPO1

        :param name: (str) type of the policy (lin, logits, value)
        :param ob_space: (Gym Space) The observation space of the environment
        :param ac_space: (Gym Space) The action space of the environment
        :param hid_size: (int) the size of the hidden layers
        :param num_hid_layers: (int) the number of hidden layers
        :param sess: (TensorFlow session) The current TensorFlow session containing the variables.
        :param reuse: (bool) If the policy is reusable or not
        :param gaussian_fixed_var: (bool) enable gaussian sampling with fixed variance, when using continuous actions
        """
        super(MlpPolicy, self).__init__()
        self.reuse = reuse
        self.name = name
        self._init(*args, **kwargs)
        self.scope = tf.get_variable_scope().name
        self.sess = sess

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True):
        """

        :param ob_space: (Gym Space) The observation space of the environment
        :param ac_space: (Gym Space) The action space of the environment
        :param hid_size: (int) the size of the hidden layers
        :param num_hid_layers: (int) the number of hidden layers
        :param gaussian_fixed_var: (bool) enable gaussian sampling with fixed variance, when using continuous actions
        """
        obs, pdtype = self.get_obs_and_pdtype(ob_space, ac_space)

        with tf.variable_scope(self.name + "/obfilter", reuse=self.reuse):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        with tf.variable_scope(self.name + '/vf', reuse=self.reuse):
            obz = tf.clip_by_value((obs - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            last_out = obz
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name="fc%i" % (i + 1),
                                                      kernel_initializer=tf_util.normc_initializer(1.0)))
            self.vpred = tf.layers.dense(last_out, 1, name='final',
                                         kernel_initializer=tf_util.normc_initializer(1.0))[:, 0]

        with tf.variable_scope(self.name + '/pol', reuse=self.reuse):
            last_out = obz
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name='fc%i' % (i + 1),
                                                      kernel_initializer=tf_util.normc_initializer(1.0)))
            if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                mean = tf.layers.dense(last_out, pdtype.param_shape()[0] // 2, name='final',
                                       kernel_initializer=tf_util.normc_initializer(0.01))
                logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0] // 2],
                                         initializer=tf.zeros_initializer())
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            else:
                pdparam = tf.layers.dense(last_out, pdtype.param_shape()[0], name='final',
                                          kernel_initializer=tf_util.normc_initializer(0.01))

        self.proba_distribution = pdtype.proba_distribution_from_flat(pdparam)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        action = tf_util.switch(stochastic, self.proba_distribution.sample(), self.proba_distribution.mode())
        self._act = tf_util.function([stochastic, obs], [action, self.vpred])
