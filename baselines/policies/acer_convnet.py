# coding: utf-8

import tensorflow as tf
# from algos.acer import Acer
from policies.agent import Agent


class AcerConvnet(Agent):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack,
                 reuse=False, name='AcerConvnet'):
        super(AcerConvnet, self).__init__(name=name)
        self.batch_size = nenv * nsteps
        self.session = sess
        self.observation_space = ob_space
        self.action_space = ac_space
        self.nstack = nstack
        self.steps = nsteps
        self.name = name
        self.reuse = reuse
        nh, nw, nc = self.observation_space.shape
        self.observation_shape = (self.batch_size, nh, nw, nc * self.nstack)
        self.nactions = self.action_space.n
        self.X = tf.placeholder(tf.uint8, self.observation_shape)  # obs
        self.setup

    @Agent.define_scope
    def setup(self):
        with tf.variable_scope(self.name, reuse=self.reuse):
            h = self.nature_cnn(self.X)
            pi_logits = self.fc(h, 'pi', self.nactions, init_scale=0.01)
            self.pi = self.activation('softmax')(pi_logits)
            self.q = self.fc(h, 'q', self.nactions)

        self.action = self.sample(pi_logits)  # could change this to use self.pi instead
        self.initial_state = []  # not stateful
        # self.X = X

    def step(self, observation, *args, **kwargs):
        # returns actions, μ, states
        a0, pi0 = self.session.run(
            [self.action, self.pi],
            {self.X: observation}
        )
        return a0, pi0, []  # dummy state

    def out(self, observation, *args, **kwargs):
        pi0, q0 = self.session.run(
            [self.pi, self.q],
            {self.X: observation}
        )
        return pi0, q0

    def act(self, observation, *args, **kwargs):
        return self.session.run(self.action, {self.X: observation})
