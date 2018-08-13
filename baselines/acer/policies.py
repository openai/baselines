import numpy as np
import tensorflow as tf
from baselines.common.policies import nature_cnn
from baselines.a2c.utils import fc, batch_to_seq, seq_to_batch, lstm, sample


class AcerCnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
        nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc * nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape)  # obs
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)
            pi_logits = fc(h, 'pi', nact, init_scale=0.01)
            pi = tf.nn.softmax(pi_logits)
            q = fc(h, 'q', nact)

        a = sample(tf.nn.softmax(pi_logits))  # could change this to use self.pi instead
        self.initial_state = []  # not stateful
        self.X = X
        self.pi = pi  # actual policy params now
        self.pi_logits = pi_logits
        self.q = q
        self.vf = q

        def step(ob, *args, **kwargs):
            # returns actions, mus, states
            a0, pi0 = sess.run([a, pi], {X: ob})
            return a0, pi0, []  # dummy state

        def out(ob, *args, **kwargs):
            pi0, q0 = sess.run([pi, q], {X: ob})
            return pi0, q0

        def act(ob, *args, **kwargs):
            return sess.run(a, {X: ob})

        self.step = step
        self.out = out
        self.act = act

class AcerLstmPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False, nlstm=256):
        nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc * nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape)  # obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)

            # lstm
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)

            pi_logits = fc(h5, 'pi', nact, init_scale=0.01)
            pi = tf.nn.softmax(pi_logits)
            q = fc(h5, 'q', nact)

        a = sample(pi_logits)  # could change this to use self.pi instead
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)
        self.X = X
        self.M = M
        self.S = S
        self.pi = pi  # actual policy params now
        self.q = q

        def step(ob, state, mask, *args, **kwargs):
            # returns actions, mus, states
            a0, pi0, s = sess.run([a, pi, snew], {X: ob, S: state, M: mask})
            return a0, pi0, s

        self.step = step
