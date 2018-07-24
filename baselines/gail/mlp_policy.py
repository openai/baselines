"""
from baselines/ppo1/mlp_policy.py and add simple modification
(1) add reuse argument
(2) cache the `stochastic` placeholder
"""
import gym
import tensorflow as tf

import baselines.common.tf_util as tf_util
from baselines.acktr.utils import dense
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.ppo1.mlp_policy import BasePolicy


class MlpPolicy(BasePolicy):
    recurrent = False

    def __init__(self, name, reuse=False, *args, **kwargs):
        """
        MLP policy for Gail

        :param name: (str) the variable scope name
        :param reuse: (bool) allow resue of the graph
        :param ob_space: (Gym Space) the observation space
        :param ac_space: (Gym Space) the action space
        :param hid_size: (int) the number of hidden neurons for every hidden layer
        :param num_hid_layers: (int) the number of hidden layers
        :param gaussian_fixed_var: (bool) fix the gaussian variance
        """
        super(MlpPolicy, self).__init__()
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True):

        obs, pdtype = self.get_obs_and_pdtype(ob_space, ac_space)

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        obz = tf.clip_by_value((obs - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(dense(last_out, hid_size, "vffc%i" % (i+1),
                                        weight_init=tf_util.normc_initializer(1.0)))
        self.vpred = dense(last_out, 1, "vffinal", weight_init=tf_util.normc_initializer(1.0))[:, 0]

        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(dense(last_out, hid_size, "polfc%i" % (i+1),
                                        weight_init=tf_util.normc_initializer(1.0)))

        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            mean = dense(last_out, pdtype.param_shape()[0] // 2, "polfinal", tf_util.normc_initializer(0.01))
            logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2],
                                     initializer=tf.zeros_initializer())
            pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        else:
            pdparam = dense(last_out, pdtype.param_shape()[0], "polfinal", tf_util.normc_initializer(0.01))

        self.proba_distribution = pdtype.proba_distribution_from_flat(pdparam)
        self.state_in = []
        self.state_out = []

        # change for BC
        self.stochastic_ph = tf.placeholder(dtype=tf.bool, shape=(), name="stochastic")
        action = tf_util.switch(self.stochastic_ph, self.proba_distribution.sample(), self.proba_distribution.mode())
        self.action = action
        self._act = tf_util.function([self.stochastic_ph, obs], [action, self.vpred])
