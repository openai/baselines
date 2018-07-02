import numpy as np
import tensorflow as tf
from baselines.ppo2.policies import nature_cnn
from baselines.a2c.utils import fc, batch_to_seq, seq_to_batch, lstm, sample


class AcerPolicy(object):
    """
    Policy object for Acer
    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (tuple) The observation space of the environment
    :param ac_space: (tuple) The action space of the environment
    :param nenv: (int) The number of environments
    :param nsteps: (int) The number of steps to run
    :param nstack: (int) The number of frames stacked
    :param reuse: (bool) If the policy is reusable or not
    :param nlstm: (int) The number of LSTM cells (for reccurent policies)
    """

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False, nlstm=256):
        self.nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        self.ob_shape = (self.nbatch, nh, nw, nc * nstack)
        self.nact = ac_space.n
        self.obs_ph = tf.placeholder(tf.uint8, self.ob_shape)  # obs
        self.masks_ph = tf.placeholder(tf.float32, [self.nbatch])  # mask (done t-1)
        self.states_ph = tf.placeholder(tf.float32, [nenv, nlstm * 2])  # states
        self.sess = sess
        self.reuse = reuse

    def step(self, obs, state, mask, *args, **kwargs):
        """
        Returns the policy for a single step
        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in reccurent policies)
        :param mask: ([float]) The last masks (used in reccurent policies)
        :param args:
        :param kwargs:
        :return: ([float], [float], [float], [float]) action, mu, states
        """
        raise NotImplementedError()

    def out(self, obs, state, mask, *args, **kwargs):
        """
        Returns the pi and q values for a single step
        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in reccurent policies)
        :param mask: ([float]) The last masks (used in reccurent policies)
        :param args:
        :param kwargs:
        :return: ([float], [float]) pi, q
        """
        raise NotImplementedError()

    def act(self, obs, state, mask, *args, **kwargs):
        """
        Returns the action for a single step
        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in reccurent policies)
        :param mask: ([float]) The last masks (used in reccurent policies)
        :param args:
        :param kwargs:
        :return: ([float]) The action
        """
        raise NotImplementedError()


class AcerCnnPolicy(AcerPolicy):
    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
        super(AcerCnnPolicy, self).__init__(sess, ob_space, ac_space, nenv, nsteps, nstack, reuse)
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(self.obs_ph)
            pi_logits = fc(h, 'pi', self.nact, init_scale=0.01)
            pi = tf.nn.softmax(pi_logits)
            q = fc(h, 'q', self.nact)

        self.a = sample(pi_logits)  # could change this to use self.pi instead
        self.initial_state = []  # not stateful
        self.pi = pi  # actual policy params now
        self.q = q

    def step(self, obs, state, mask, *args, **kwargs):
        # returns actions, mus, states
        a0, pi0 = self.sess.run([self.a, self.pi], {self.obs_ph: obs})
        return a0, pi0, []  # dummy state

    def out(self, obs, state, mask, *args, **kwargs):
        pi0, q0 = self.sess.run([self.pi, self.q], {self.obs_ph: obs})
        return pi0, q0

    def act(self, obs, state, mask, *args, **kwargs):
        return self.sess.run(self.a, {self.obs_ph: obs})


class AcerLstmPolicy(AcerPolicy):
    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False, nlstm=256):
        super(AcerLstmPolicy, self).__init__(sess, ob_space, ac_space, nenv, nsteps, nstack, reuse, nlstm)
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(self.obs_ph)

            # lstm
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(self.masks_ph, nenv, nsteps)
            h5, self.snew = lstm(xs, ms, self.states_ph, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)

            pi_logits = fc(h5, 'pi', self.nact, init_scale=0.01)
            pi = tf.nn.softmax(pi_logits)
            q = fc(h5, 'q', self.nact)

        self.a = sample(pi_logits)  # could change this to use self.pi instead
        self.initial_state = np.zeros((nenv, nlstm * 2), dtype=np.float32)
        self.pi = pi  # actual policy params now
        self.q = q

    def step(self, obs, state, mask, *args, **kwargs):
        # returns actions, mus, states
        a0, pi0, s = self.sess.run([self.a, self.pi, self.snew],
                                   {self.obs_ph: obs, self.states_ph: state, self.masks_ph: mask})
        return a0, pi0, s

    def out(self, obs, state, mask, *args, **kwargs):
        pi0, q0 = self.sess.run([self.pi, self.q], {self.obs_ph: obs})
        return pi0, q0

    def act(self, obs, state, mask, *args, **kwargs):
        return self.sess.run(self.a, {self.obs_ph: obs})
