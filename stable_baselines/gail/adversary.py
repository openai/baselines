"""
Reference: https://github.com/openai/imitation
I follow the architecture from the official repository
"""
import tensorflow as tf
import numpy as np

from stable_baselines.common.mpi_running_mean_std import RunningMeanStd
from stable_baselines.common import tf_util as tf_util


def logsigmoid(input_tensor):
    """
    Equivalent to tf.log(tf.sigmoid(a))

    :param input_tensor: (TensorFlow Tensor)
    :return: (TensorFlow Tensor)
    """
    return -tf.nn.softplus(-input_tensor)


def logit_bernoulli_entropy(logits):
    """
    Reference:
    https://github.com/openai/imitation/blob/99fbccf3e060b6e6c739bdf209758620fcdefd3c/policyopt/thutil.py#L48-L51

    :param logits: (TensorFlow Tensor) the logits
    :return: (TensorFlow Tensor) the bernoulli entropy
    """
    ent = (1. - tf.nn.sigmoid(logits)) * logits - logsigmoid(logits)
    return ent


class TransitionClassifier(object):
    def __init__(self, env, hidden_size, entcoeff=0.001, scope="adversary"):
        """
        reward regression from observations and transitions

        :param env: (Gym Environment)
        :param hidden_size: ([int]) the hidden dimension for the MLP
        :param entcoeff: (float) the entropy loss weight
        :param scope: (str) tensorflow variable scope
        """
        self.scope = scope
        self.observation_shape = env.observation_space.shape
        self.actions_shape = env.action_space.shape
        self.input_shape = tuple([o + a for o, a in zip(self.observation_shape, self.actions_shape)])
        self.num_actions = env.action_space.shape[0]
        self.hidden_size = hidden_size
        self.build_ph()
        # Build grpah
        generator_logits = self.build_graph(self.generator_obs_ph, self.generator_acs_ph, reuse=False)
        expert_logits = self.build_graph(self.expert_obs_ph, self.expert_acs_ph, reuse=True)
        # Build accuracy
        generator_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(generator_logits) < 0.5))
        expert_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(expert_logits) > 0.5))
        # Build regression loss
        # let x = logits, z = targets.
        # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=generator_logits,
                                                                 labels=tf.zeros_like(generator_logits))
        generator_loss = tf.reduce_mean(generator_loss)
        expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=expert_logits, labels=tf.ones_like(expert_logits))
        expert_loss = tf.reduce_mean(expert_loss)
        # Build entropy loss
        logits = tf.concat([generator_logits, expert_logits], 0)
        entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
        entropy_loss = -entcoeff * entropy
        # Loss + Accuracy terms
        self.losses = [generator_loss, expert_loss, entropy, entropy_loss, generator_acc, expert_acc]
        self.loss_name = ["generator_loss", "expert_loss", "entropy", "entropy_loss", "generator_acc", "expert_acc"]
        self.total_loss = generator_loss + expert_loss + entropy_loss
        # Build Reward for policy
        self.reward_op = -tf.log(1 - tf.nn.sigmoid(generator_logits) + 1e-8)
        var_list = self.get_trainable_variables()
        self.lossandgrad = tf_util.function(
            [self.generator_obs_ph, self.generator_acs_ph, self.expert_obs_ph, self.expert_acs_ph],
            self.losses + [tf_util.flatgrad(self.total_loss, var_list)])

    def build_ph(self):
        """
        build placeholder
        """
        self.generator_obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape,
                                               name="observations_ph")
        self.generator_acs_ph = tf.placeholder(tf.float32, (None,) + self.actions_shape,
                                               name="actions_ph")
        self.expert_obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape,
                                            name="expert_observations_ph")
        self.expert_acs_ph = tf.placeholder(tf.float32, (None,) + self.actions_shape,
                                            name="expert_actions_ph")

    def build_graph(self, obs_ph, acs_ph, reuse=False):
        """
        build the graph

        :param obs_ph: (TensorFlow Tensor) the observation placeholder
        :param acs_ph: (TensorFlow Tensor) the action placeholder
        :param reuse: (bool)
        :return: (TensorFlow Tensor) the graph output
        """
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("obfilter"):
                self.obs_rms = RunningMeanStd(shape=self.observation_shape)
            obs = (obs_ph - self.obs_rms.mean / self.obs_rms.std)
            _input = tf.concat([obs, acs_ph], axis=1)  # concatenate the two input -> form a transition
            p_h1 = tf.contrib.layers.fully_connected(_input, self.hidden_size, activation_fn=tf.nn.tanh)
            p_h2 = tf.contrib.layers.fully_connected(p_h1, self.hidden_size, activation_fn=tf.nn.tanh)
            logits = tf.contrib.layers.fully_connected(p_h2, 1, activation_fn=tf.identity)
        return logits

    def get_trainable_variables(self):
        """
        get all the trainable variables from the graph

        :return: ([TensorFlow Tensor]) the variables
        """
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_reward(self, obs, actions):
        """
        get the reward using the observation and action

        :param obs: (TensorFlow Tensor or np.ndarray) the observation
        :param actions: (TensorFlow Tensor or np.ndarray) the action
        :return: (np.ndarray) the reward
        """
        sess = tf.get_default_session()
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)
        if len(actions.shape) == 1:
            actions = np.expand_dims(actions, 0)
        feed_dict = {self.generator_obs_ph: obs, self.generator_acs_ph: actions}
        reward = sess.run(self.reward_op, feed_dict)
        return reward
