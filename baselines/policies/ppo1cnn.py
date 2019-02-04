import gym
import tensorflow as tf
from policies.agent import Agent
from utils.distributions import make_pdtype


class PPO1Cnn(Agent):
    recurrent = False

    def __init__(self, name, ob_space, ac_space, kind='large'):
        super(PPO1Cnn, self).__init__(name='PPO1Cnn')
        with tf.variable_scope(name):
            self._init(ob_space, ac_space, kind)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, kind):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = self.get_placeholder(
            name="ob",
            dtype=tf.float32,
            shape=[sequence_length] + list(ob_space.shape)
        )

        x = ob / 255.0
        if kind == 'small': # from A3C paper
            x = self.activation('relu')(
                self.conv2d(x, 16, "l1", [8, 8], [4, 4], pad="VALID")
            )
            x = self.activation('relu')(
                self.conv2d(x, 32, "l2", [4, 4], [2, 2], pad="VALID")
            )
            x = self.flatten_except_first(x)
            x = self.activation('relu')(
                self.dense(inputs=x, units=256, name='lin',
                           kernel_initializer=self.normc_initializer(1.0)))
        elif kind == 'large': # Nature DQN
            x = self.activation('relu')(
                self.conv2d(x, 32, "l1", [8, 8], [4, 4], pad="VALID")
            )
            x = self.activation('relu')(
                self.conv2d(x, 64, "l2", [4, 4], [2, 2], pad="VALID")
            )
            x = self.activation('relu')(
                self.conv2d(x, 64, "l3", [3, 3], [1, 1], pad="VALID")
            )
            x = self.flatten_except_first(x)
            x = self.activation('relu')(
                self.dense(inputs=x, units=512, name='lin',
                           kernel_initializer=self.normc_initializer(1.0))
            )
        else:
            raise NotImplementedError

        logits = self.dense(inputs=x, units=pdtype.param_shape()[0], name='logits',
                            kernel_initializer=self.normc_initializer(0.01))
        self.pd = pdtype.pdfromflat(logits)
        self.vpred = self.dense(
            inputs=x, units=1, name='value',
            kernel_initializer=self.normc_initializer(1.0))[:, 0]

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = self.pd.sample() # XXX
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
