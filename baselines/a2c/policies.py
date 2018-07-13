import numpy as np
import tensorflow as tf

from baselines.a2c.utils import conv, linear, conv_to_fc, batch_to_seq, seq_to_batch, lstm
from baselines.common.distributions import make_proba_dist_type
from baselines.common.input import observation_input


def nature_cnn(unscaled_images, **kwargs):
    """
    CNN from Nature paper.

    :param unscaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_3)
    return activ(linear(layer_3, 'fc1', nh=512, init_scale=np.sqrt(2), **kwargs))


class A2CPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        """
        Policy object for A2C

        :param sess: (TensorFlow session) The current TensorFlow session
        :param ob_space: (Gym Space) The observation space of the environment
        :param ac_space: (Gym Space) The action space of the environment
        :param nbatch: (int) The number of batch to run (nenvs * nsteps)
        :param nsteps: (int) The number of steps to run for each environment
        :param nlstm: (int) The number of LSTM cells (for reccurent policies)
        :param reuse: (bool) If the policy is reusable or not
        """
        self.nenv = nbatch // nsteps
        self.obs_ph, self.processed_x = observation_input(ob_space, nbatch)
        self.masks_ph = tf.placeholder(tf.float32, [nbatch])  # mask (done t-1)
        self.states_ph = tf.placeholder(tf.float32, [self.nenv, nlstm * 2])  # states
        self.pdtype = make_proba_dist_type(ac_space)
        self.sess = sess
        self.reuse = reuse

    def step(self, obs, state=None, mask=None):
        """
        Returns the policy for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in reccurent policies)
        :param mask: ([float]) The last masks (used in reccurent policies)
        :return: ([float], [float], [float], [float]) actions, values, states, neglogp0
        """
        raise NotImplementedError()

    def value(self, obs, state=None, mask=None):
        """
        Returns the value for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in reccurent policies)
        :param mask: ([float]) The last masks (used in reccurent policies)
        :return: ([float]) The associated value of the action
        """
        raise NotImplementedError()


class LstmPolicy(A2CPolicy):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False, layer_norm=False, **kwargs):
        super(LstmPolicy, self).__init__(sess, ob_space, ac_space, nbatch, nsteps, nlstm, reuse)
        with tf.variable_scope("model", reuse=reuse):
            extracted_features = nature_cnn(self.obs_ph, **kwargs)
            input_sequence = batch_to_seq(extracted_features, self.nenv, nsteps)
            masks = batch_to_seq(self.masks_ph, self.nenv, nsteps)
            rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=nlstm,
                                         layer_norm=layer_norm)
            rnn_output = seq_to_batch(rnn_output)
            value_fn = linear(rnn_output, 'v', 1)
            self.proba_distribution, self.policy = self.pdtype.probability_distribution_from_latent(rnn_output)

        self.value_0 = value_fn[:, 0]
        self.action_0 = self.proba_distribution.sample()
        self.neglogp0 = self.proba_distribution.neglogp(self.action_0)
        self.initial_state = np.zeros((self.nenv, nlstm * 2), dtype=np.float32)
        self.value_fn = value_fn

    def step(self, obs, state=None, mask=None):
        return self.sess.run([self.action_0, self.value_0, self.snew, self.neglogp0],
                             {self.obs_ph: obs, self.states_ph: state, self.masks_ph: mask})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_0, {self.obs_ph: obs, self.states_ph: state, self.masks_ph: mask})


class LnLstmPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False, **_):
        super(LnLstmPolicy, self).__init__(sess, ob_space, ac_space, nbatch, nsteps, nlstm, reuse, layer_norm=True)


class FeedForwardPolicy(A2CPolicy):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False, _type="cnn", **kwargs):
        super(FeedForwardPolicy, self).__init__(sess, ob_space, ac_space, nbatch, nsteps, nlstm, reuse)
        with tf.variable_scope("model", reuse=reuse):
            if _type == "cnn":
                extracted_features = nature_cnn(self.processed_x, **kwargs)
                value_fn = linear(extracted_features, 'v', 1)[:, 0]
            else:
                activ = tf.tanh
                processed_x = tf.layers.flatten(self.processed_x)
                pi_h1 = activ(linear(processed_x, 'pi_fc1', nh=64, init_scale=np.sqrt(2), **kwargs))
                pi_h2 = activ(linear(pi_h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2), **kwargs))
                vf_h1 = activ(linear(processed_x, 'vf_fc1', nh=64, init_scale=np.sqrt(2), **kwargs))
                vf_h2 = activ(linear(vf_h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2), **kwargs))
                value_fn = linear(vf_h2, 'vf', 1)[:, 0]
                extracted_features = pi_h2
            self.proba_distribution, self.policy = self.pdtype.probability_distribution_from_latent(extracted_features, init_scale=0.01)

        self.action_0 = self.proba_distribution.sample()
        self.neglogp0 = self.proba_distribution.neglogp(self.action_0)
        self.initial_state = None
        self.value_fn = value_fn

    def step(self, obs, state=None, mask=None):
        action, value, neglogp = self.sess.run([self.action_0, self.value_fn, self.neglogp0], {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_fn, {self.obs_ph: obs})


class CnnPolicy(FeedForwardPolicy):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False, **_kwargs):
        super(CnnPolicy, self).__init__(sess, ob_space, ac_space, nbatch, nsteps, nlstm, reuse, _type="cnn")


class MlpPolicy(FeedForwardPolicy):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False, **_kwargs):
        super(MlpPolicy, self).__init__(sess, ob_space, ac_space, nbatch, nsteps, nlstm, reuse, _type="mlp")
