import tensorflow as tf
import numpy as np
from gym.spaces import Box

from stable_baselines.a2c.utils import linear
from stable_baselines.common.distributions import make_proba_dist_type
from stable_baselines.common.policies import BasePolicy, nature_cnn


class DeepQPolicy(BasePolicy):
    """
    Policy object that implements a DQN policy

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for reccurent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param scale: (bool) whether or not to scale the input
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, scale=False):
        super(DeepQPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=n_lstm, reuse=reuse,
                                          scale=scale)
        assert not isinstance(ac_space, Box), "Error: the action space cannot be of type gym.spaces.Box"
        self.pdtype = make_proba_dist_type(ac_space)
        self.value_fn = None
        self.proba_distribution = None
        self.policy = None

    def _setup_init(self):
        """
        sets up the distibutions, actions, and value
        """
        with tf.variable_scope("output", reuse=True):
            assert self.policy is not None and self.proba_distribution is not None and self.value_fn is not None
            self.action = self.proba_distribution.sample()
            self.neglogp = self.proba_distribution.neglogp(self.action)
            self.policy_proba = tf.nn.softmax(self.policy)
            self._value = self.value_fn[:, 0]

    def step(self, obs, state=None, mask=None):
        """
        Returns the policy for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in reccurent policies)
        :param mask: ([float]) The last masks (used in reccurent policies)
        :return: ([float], [float], [float], [float]) actions, values, states, neglogp
        """
        raise NotImplementedError

    def proba_step(self, obs, state=None, mask=None):
        """
        Returns the action probability for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in reccurent policies)
        :param mask: ([float]) The last masks (used in reccurent policies)
        :return: ([float]) the action probability
        """
        raise NotImplementedError

    def value(self, obs, state=None, mask=None):
        """
        Returns the value for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in reccurent policies)
        :param mask: ([float]) The last masks (used in reccurent policies)
        :return: ([float]) The associated value of the action
        """
        raise NotImplementedError


class FeedForwardPolicy(DeepQPolicy):
    """
    Policy object that implements a DQN policy, using a feed forward neural network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network for the policy (if None, default to [64, 64])
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None,
                 cnn_extractor=nature_cnn, feature_extraction="cnn", **kwargs):
        super(FeedForwardPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256,
                                                reuse=reuse, scale=(feature_extraction == "cnn"))
        if layers is None:
            layers = [64, 64]

        with tf.variable_scope("model", reuse=reuse):
            if feature_extraction == "cnn":
                extracted_features = cnn_extractor(self.processed_x, **kwargs)
                pi_latent = extracted_features
            else:
                activ = tf.tanh
                processed_x = tf.layers.flatten(self.processed_x)
                pi_h = processed_x
                for i, layer_size in enumerate(layers):
                    pi_h = activ(linear(pi_h, 'pi_fc' + str(i), n_hidden=layer_size, init_scale=np.sqrt(2)))
                pi_latent = pi_h

            self.proba_distribution, self.policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, pi_latent, init_scale=0.01)

        self.value_fn = pi_latent
        self.initial_state = None
        self._setup_init()

    def step(self, obs, state=None, mask=None):
        action, value = self.sess.run([self.policy, self._value], {self.obs_ph: obs})
        return action, value, None

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self._value, {self.obs_ph: obs})


class CnnPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a CNN (the nature CNN)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(CnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="cnn", **_kwargs)


class MlpPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a MLP (2 layers of 64)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(MlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="mlp", **_kwargs)


def _mlp(hiddens, inpt, num_actions, scope, reuse=False, layer_norm=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        for hidden in hiddens:
            out = layers.fully_connected(out, num_outputs=hidden, activation_fn=None)
            if layer_norm:
                out = layers.layer_norm(out, center=True, scale=True)
            out = tf.nn.relu(out)
        q_out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return q_out


def mlp(hiddens=None, layer_norm=False):
    """
    This model takes as input an observation and returns values of all actions.

    :param hiddens: ([int]) list of sizes of hidden layers
    :param layer_norm: (bool) if true, use layer normalization

    :return: (function) q_function for DQN algorithm.
    """
    if hiddens is None:
        hiddens = []
    return lambda *args, **kwargs: _mlp(hiddens, layer_norm=layer_norm, *args, **kwargs)


def _cnn_to_mlp(convs, hiddens, dueling, inpt, num_actions, scope, reuse=False, layer_norm=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        with tf.variable_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                out = layers.convolution2d(out,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=tf.nn.relu)
        conv_out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            action_out = conv_out
            for hidden in hiddens:
                action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=None)
                if layer_norm:
                    action_out = layers.layer_norm(action_out, center=True, scale=True)
                action_out = tf.nn.relu(action_out)
            action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)

        if dueling:
            with tf.variable_scope("state_value"):
                state_out = conv_out
                for hidden in hiddens:
                    state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=None)
                    if layer_norm:
                        state_out = layers.layer_norm(state_out, center=True, scale=True)
                    state_out = tf.nn.relu(state_out)
                state_score = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
            q_out = state_score + action_scores_centered
        else:
            q_out = action_scores
        return q_out


def cnn_to_mlp(convs, hiddens, dueling=False, layer_norm=False):
    """This model takes as input an observation and returns values of all actions.

    :param convs: ([(int, int, int)]) list of convolutional layers in form of (num_outputs, kernel_size, stride)
    :param hiddens: ([int]) list of sizes of hidden layers
    :param dueling: (bool) if true double the output MLP to compute a baseline for action scores
    :param layer_norm: (bool) if true, use layer normalization
    :return: (function) q_function for DQN algorithm.
    """

    return lambda *args, **kwargs: _cnn_to_mlp(convs, hiddens, dueling, layer_norm=layer_norm, *args, **kwargs)

