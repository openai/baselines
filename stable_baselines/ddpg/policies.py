import tensorflow as tf
import numpy as np
from gym.spaces import Box

from stable_baselines.common.policies import BasePolicy, nature_cnn
from stable_baselines.a2c.utils import linear


class DDPGPolicy(BasePolicy):
    """
    Policy object that implements a DDPG-like actor critic

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
        super(DDPGPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=n_lstm, reuse=reuse,
                                         scale=scale)
        assert isinstance(ac_space, Box), "Error: the action space must be of type gym.spaces.Box"
        assert np.abs(ac_space.low) == ac_space.high, "Error: the action space low and high must be symetric"
        self.value_fn = None
        self.policy = None

    def step(self, obs, state=None, mask=None):
        """
        Returns the policy for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in reccurent policies)
        :param mask: ([float]) The last masks (used in reccurent policies)
        :return: ([float], [float], [float]) actions, values, states
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


class FeedForwardPolicy(DDPGPolicy):
    """
    Policy object that implements a DDPG-like actor critic, using a feed forward neural network.

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
                processed_x = cnn_extractor(self.processed_x, **kwargs)
            else:
                processed_x = tf.layers.flatten(self.processed_x)

            activ = tf.tanh
            pi_h = processed_x
            for i, layer_size in enumerate(layers):
                pi_h = activ(linear(pi_h, 'pi_fc' + str(i), n_hidden=layer_size, init_scale=np.sqrt(2)))
            pi_latent = tf.tanh(linear(pi_h, 'pi_fc_out', n_hidden=self.ac_space.shape[0], init_scale=np.sqrt(2)))

            vf_h = tf.concat([processed_x, pi_latent], axis=-1)
            for i, layer_size in enumerate(layers):
                vf_h = activ(linear(vf_h, 'vf_fc' + str(i), n_hidden=layer_size, init_scale=np.sqrt(2)))
            value_fn = linear(vf_h, 'vf', 1)

            self.policy = tf.multiply(pi_latent, tf.convert_to_tensor(np.abs(self.ac_space.low)))

        self.value_fn = value_fn
        self._value = value_fn[:, 0]

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
