import gym
import tensorflow as tf
from policies.agent import Agent
from utils.distributions import make_pdtype


class CnnPolicy(Agent):
    recurrent = False

    def __init__(self, name, ob_space, ac_space):
        super(CnnPolicy, self).__init__()
        with tf.variable_scope(name):
            self._init(ob_space, ac_space)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = self.get_placeholder(
            name="ob",
            dtype=tf.float32,
            shape=[sequence_length] + list(ob_space.shape)
        )

        obscaled = ob / 255.0

        with tf.variable_scope("pol"):
            x = obscaled
            x = self.activation('relu')(
                self.conv2d(x, 8, "l1", [8, 8], [4, 4], pad="VALID")
            )
            x = self.activation('relu')(
                self.conv2d(x, 16, "l2", [4, 4], [2, 2], pad="VALID")
            )
            x = self.flatten_except_first(x)
            x = self.activation('relu')(
                self.dense(inputs=x, units=128, name='lin',
                           kernel_initializer=self.normc_initializer(1.0))
            )
            logits = self.dense(
                inputs=x,
                units=pdtype.param_shape()[0],
                name='logits',
                kernel_initializer=self.normc_initializer(0.01))
            self.pd = pdtype.pdfromflat(logits)
        with tf.variable_scope("vf"):
            x = obscaled
            x = self.activation('relu')(
                self.conv2d(x, 8, "l1", [8, 8], [4, 4], pad="VALID")
            )
            x = self.activation('relu')(
                self.conv2d(x, 16, "l2", [4, 4], [2, 2], pad="VALID")
            )
            x = self.flatten_except_first(x)
            x = self.activation('relu')(
                self.dense(
                    inputs=x,
                    units=128,
                    name='lin',
                    kernel_initializer=self.normc_initializer(1.0))
            )
            self.vpred = self.dense(
                inputs=x,
                units=1,
                name='value',
                kernel_initializer=self.normc_initializer(1.0))
            self.vpredz = self.vpred

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = self.pd.sample()
        self._act = self.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1 = self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []
