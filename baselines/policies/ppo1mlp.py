import gym
import tensorflow as tf
from policies.model import Model
from utils.distributions import make_pdtype
from mpi.mpi_running_mean_std import RunningMeanStd


class PPO1Mlp(Model):
    recurrent = False

    def __init__(self, name, *args, **kwargs):
        super(PPO1Mlp, self).__init__(name='PPO1Mlp')

        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers,
              gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = self.get_placeholder(
            name="ob",
            dtype=tf.float32,
            shape=[sequence_length] + list(ob_space.shape)
        )

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        with tf.variable_scope('vf'):
            obz = tf.clip_by_value(
                (ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0
            )
            last_out = obz
            for i in range(num_hid_layers):
                last_out = self.activation('tanh')(
                    self.dense(inputs=last_out, units=hid_size, name="fc%i"%(i+1),
                               kernel_initializer=self.normc_initializer(1.0))
                )
            self.vpred = self.dense(
                inputs=last_out,
                units=1, name='final',
                kernel_initializer=self.normc_initializer(1.0)
                )[:, 0]

        with tf.variable_scope('pol'):
            last_out = obz
            for i in range(num_hid_layers):
                last_out = self.activation('tanh')(
                    self.dense(inputs=last_out, units=hid_size, name='fc%i'%(i+1),
                               kernel_initializer=self.normc_initializer(1.0))
                )
            if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                mean = self.dense(inputs=last_out, units=pdtype.param_shape()[0]//2,
                                  name='final',
                                  kernel_initializer=self.normc_initializer(0.01))
                logstd = tf.get_variable(
                    name="logstd",
                    shape=[1, pdtype.param_shape()[0]//2],
                    initializer=tf.zeros_initializer()
                )
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            else:
                pdparam = self.dense(inputs=last_out, units=pdtype.param_shape()[0],
                                     name='final',
                                     kernel_initializer=U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = self.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = self.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1 =  self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []
