import numpy as np
import tensorflow as tf
from stable_baselines.a2c.policies import nature_cnn
from stable_baselines.a2c.utils import linear, batch_to_seq, seq_to_batch, lstm, sample


class AcerPolicy(object):
    """
    Policy object for Acer
    
    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments
    :param n_steps: (int) The number of steps to run
    :param n_stack: (int) The number of frames stacked
    :param reuse: (bool) If the policy is reusable or not
    :param n_lstm: (int) The number of LSTM cells (for reccurent policies)
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_stack, reuse=False, n_lstm=256):
        self.n_batch = n_env * n_steps
        height, width, n_channels = ob_space.shape
        self.ob_shape = (self.n_batch, height, width, n_channels * n_stack)
        self.n_act = ac_space.n
        self.obs_ph = tf.placeholder(tf.uint8, self.ob_shape)  # obs
        self.masks_ph = tf.placeholder(tf.float32, [self.n_batch])  # mask (done t-1)
        self.states_ph = tf.placeholder(tf.float32, [n_env, n_lstm * 2])  # states
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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError


class AcerCnnPolicy(AcerPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_stack, reuse=False):
        super(AcerCnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_stack, reuse)
        with tf.variable_scope("model", reuse=reuse):
            extracted_features = nature_cnn(self.obs_ph)
            pi_logits = linear(extracted_features, 'pi', self.n_act, init_scale=0.01)
            policy = tf.nn.softmax(pi_logits)
            q_value = linear(extracted_features, 'q', self.n_act)

        self.action = sample(pi_logits)  # could change this to use self.pi instead
        self.initial_state = []  # not stateful
        self.policy = policy  # actual policy params now
        self.q_value = q_value

    def step(self, obs, state, mask, *args, **kwargs):
        # returns actions, mus, states
        action, policy = self.sess.run([self.action, self.policy], {self.obs_ph: obs})
        return action, policy, []  # dummy state

    def out(self, obs, state, mask, *args, **kwargs):
        policy, q_value = self.sess.run([self.policy, self.q_value], {self.obs_ph: obs})
        return policy, q_value

    def act(self, obs, state, mask, *args, **kwargs):
        return self.sess.run(self.action, {self.obs_ph: obs})


class AcerLstmPolicy(AcerPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_stack, reuse=False, n_lstm=256):
        super(AcerLstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_stack, reuse, n_lstm)
        with tf.variable_scope("model", reuse=reuse):
            extracted_features = nature_cnn(self.obs_ph)

            # lstm
            input_seq = batch_to_seq(extracted_features, n_env, n_steps)
            masks = batch_to_seq(self.masks_ph, n_env, n_steps)
            rnn_output, self.snew = lstm(input_seq, masks, self.states_ph, 'lstm1', n_hidden=n_lstm)
            rnn_output = seq_to_batch(rnn_output)

            pi_logits = linear(rnn_output, 'pi', self.n_act, init_scale=0.01)
            policy = tf.nn.softmax(pi_logits)
            q_value = linear(rnn_output, 'q', self.n_act)

        self.action = sample(pi_logits)  # could change this to use self.pi instead
        self.initial_state = np.zeros((n_env, n_lstm * 2), dtype=np.float32)
        self.policy = policy  # actual policy params now
        self.q_value = q_value

    def step(self, obs, state, mask, *args, **kwargs):
        # returns actions, mus, states
        action, policy, states = self.sess.run([self.action, self.policy, self.snew],
                                   {self.obs_ph: obs, self.states_ph: state, self.masks_ph: mask})
        return action, policy, states

    def out(self, obs, state, mask, *args, **kwargs):
        policy, q_value = self.sess.run([self.policy, self.q_value], {self.obs_ph: obs})
        return policy, q_value

    def act(self, obs, state, mask, *args, **kwargs):
        return self.sess.run(self.action, {self.obs_ph: obs})
