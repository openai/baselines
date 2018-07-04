import numpy as np
import tensorflow as tf

from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype
from baselines.common.input import observation_input


def nature_cnn(unscaled_images):
    """
    CNN from Nature paper.
    """
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2)))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2)))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2)))
    h3 = conv_to_fc(h3)
    return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))


class A2CPolicy(object):
    """
    Policy object for A2C

    Parameters
    ----------
    sess: TensorFlow session
        The current TensorFlow session
    ob_space: tuple
        The observation space of the environment
    ac_space: tuple
        The action space of the environment
    nbatch: int
        The number of batch to run
    nsteps: int
        The number of steps to run
    nlstm: int
        The number of LSTM cells (for reccurent policies)
    reuse: bool
        If the policy is reusable or not
    """
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        self.nenv = nbatch // nsteps
        self.obs_ph, self.processed_x = observation_input(ob_space, nbatch)
        self.masks_ph = tf.placeholder(tf.float32, [nbatch])  # mask (done t-1)
        self.states_ph = tf.placeholder(tf.float32, [self.nenv, nlstm * 2])  # states
        self.pdtype = make_pdtype(ac_space)
        self.sess = sess
        self.reuse = reuse

    def step(self, obs, state, mask):
        """
        Returns the policy for a single step

        Parameters
        ----------
        obs: [float] or [int]
            The current observation of the environment
        state: [float]
            The last states (used in reccurent policies)
        mask: [float]
            The last masks (used in reccurent policies)

        Returns
        -------
        action: [float]
            The returned action
        value: [float]
            The associated value of the action
        states: [float]
            The updated states
        neglogp0: [float]
        """
        raise NotImplementedError()

    def value(self, obs, state, mask):
        """
        Returns the value for a single step

        Parameters
        ----------
        obs: [float] or [int]
            The current observation of the environment
        state: [float]
            The last states (used in reccurent policies)
        mask: [float]
            The last masks (used in reccurent policies)

        Returns
        -------
        value: [float]
            The associated value of the action
        """
        raise NotImplementedError()


class LnLstmPolicy(A2CPolicy):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        super(LnLstmPolicy, self).__init__(sess, ob_space, ac_space, nbatch, nsteps, nlstm, reuse)
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(self.processed_x)
            xs = batch_to_seq(h, self.nenv, nsteps)
            ms = batch_to_seq(self.masks_ph, self.nenv, nsteps)
            h5, self.snew = lnlstm(xs, ms, self.states_ph, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            vf = fc(h5, 'v', 1)
            self.pd, self.pi = self.pdtype.pdfromlatent(h5)

        self.v0 = vf[:, 0]
        self.a0 = self.pd.sample()
        self.neglogp0 = self.pd.neglogp(self.a0)
        self.initial_state = np.zeros((self.nenv, nlstm * 2), dtype=np.float32)
        self.vf = vf

    def step(self, obs, state, mask):
        return self.sess.run([self.a0, self.v0, self.snew, self.neglogp0],
                             {self.obs_ph: obs, self.states_ph: state, self.masks_ph: mask})

    def value(self, obs, state, mask):
        return self.sess.run(self.v0, {self.obs_ph: obs, self.states_ph: state, self.masks_ph: mask})


class LstmPolicy(A2CPolicy):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        super(LstmPolicy, self).__init__(sess, ob_space, ac_space, nbatch, nsteps, nlstm, reuse)
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(self.obs_ph)
            xs = batch_to_seq(h, self.nenv, nsteps)
            ms = batch_to_seq(self.masks_ph, self.nenv, nsteps)
            h5, self.snew = lstm(xs, ms, self.states_ph, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            vf = fc(h5, 'v', 1)
            self.pd, self.pi = self.pdtype.pdfromlatent(h5)

        self.v0 = vf[:, 0]
        self.a0 = self.pd.sample()
        self.neglogp0 = self.pd.neglogp(self.a0)
        self.initial_state = np.zeros((self.nenv, nlstm * 2), dtype=np.float32)
        self.vf = vf

    def step(self, obs, state, mask):
        return self.sess.run([self.a0, self.v0, self.snew, self.neglogp0],
                             {self.obs_ph: obs, self.states_ph: state, self.masks_ph: mask})

    def value(self, obs, state, mask):
        return self.sess.run(self.v0, {self.obs_ph: obs, self.states_ph: state, self.masks_ph: mask})


class CnnPolicy(A2CPolicy):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        super(CnnPolicy, self).__init__(sess, ob_space, ac_space, nbatch, nsteps, nlstm, reuse)
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(self.processed_x)
            vf = fc(h, 'v', 1)[:, 0]
            self.pd, self.pi = self.pdtype.pdfromlatent(h, init_scale=0.01)

        self.a0 = self.pd.sample()
        self.neglogp0 = self.pd.neglogp(self.a0)
        self.initial_state = None
        self.vf = vf

    def step(self, obs, *args, **kwargs):
        a, v, neglogp = self.sess.run([self.a0, self.vf, self.neglogp0], {self.obs_ph: obs})
        return a, v, self.initial_state, neglogp

    def value(self, obs, *args, **kwargs):
        return self.sess.run(self.vf, {self.obs_ph: obs})


class MlpPolicy(A2CPolicy):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        super(MlpPolicy, self).__init__(sess, ob_space, ac_space, nbatch, nsteps, nlstm, reuse)
        with tf.variable_scope("model", reuse=reuse):
            activ = tf.tanh
            processed_x = tf.layers.flatten(self.processed_x)
            pi_h1 = activ(fc(processed_x, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            pi_h2 = activ(fc(pi_h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            vf_h1 = activ(fc(processed_x, 'vf_fc1', nh=64, init_scale=np.sqrt(2)))
            vf_h2 = activ(fc(vf_h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
            vf = fc(vf_h2, 'vf', 1)[:, 0]

            self.pd, self.pi = self.pdtype.pdfromlatent(pi_h2, init_scale=0.01)

        self.a0 = self.pd.sample()
        self.neglogp0 = self.pd.neglogp(self.a0)
        self.initial_state = None
        self.vf = vf

    def step(self, obs, *args, **kwargs):
        a, v, neglogp = self.sess.run([self.a0, self.vf, self.neglogp0], {self.obs_ph: obs})
        return a, v, self.initial_state, neglogp

    def value(self, obs, *args, **kwargs):
        return self.sess.run(self.vf, {self.obs_ph: obs})

