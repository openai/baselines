import numpy as np
import tensorflow as tf
from algos.acer import Acer
from policies.model import Model


class AcerLstm(Acer):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack,
                 reuse=False, nlstm=256, name='AcerLstm'):
        self.session = sess
        self.batch_size = nenv * nsteps
        self.name = name
        self.reuse = reuse
        self.nenv = nenv
        self.nsteps = nsteps
        self.nlstm = nlstm
        nh, nw, nc = ob_space.shape
        self.observation_shape = (self.batch_size, nh, nw, nc * nstack)
        self.nactions = ac_space.n
        self.X = tf.placeholder(tf.uint8, self.observation_shape)   # observation
        self.M = tf.placeholder(tf.float32, [self.batch_size])  # mask (done t-1)
        self.S = tf.placeholder(tf.float32, [nenv, nlstm * 2])  # states

    @Model.define_scope
    def setup(self):
        with tf.variable_scope(self.name, reuse=self.reuse):
            h = self.nature_cnn(self.X)

            # lstm
            xs = self.batch_to_seq(h, self.nenv, self.nsteps)
            ms = self.batch_to_seq(self.M, self.nenv, self.nsteps)
            h5, snew = self.lstm(xs, ms, self.S, 'lstm1', nh=self.nlstm)
            self.snew = snew
            h5 = self.seq_to_batch(h5)

            pi_logits = self.fc(h5, 'pi', self.nactions, init_scale=0.01)
            self.pi = self.activation('softmax')(pi_logits)
            q = self.fc(h5, 'q', self.nactions)

        self.action = self.sample(pi_logits)  # could change this to use self.pi instead
        self.initial_state = np.zeros((self.nenv, self.nlstm*2), dtype=np.float32)
        # self.X = X
        # self.M = M
        # self.S = S
        self.pi = pi  # actual policy params now
        self.q = q

    def step(self, ob, state, mask, *args, **kwargs):
        # returns actions, mus, states
        a0, pi0, s = self.session.run([self.action, self.pi, self.snew],
                                      {self.X: ob,
                                       self.S: state,
                                       self.M: mask})
        return a0, pi0, s
