import tensorflow as tf
from baselines.a2c.utils import fc
from baselines.common.distributions import make_pdtype

import gym


class PolicyWithValue(tf.Module):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, ac_space, policy_network, value_network=None, estimate_q=False):
        """
        Parameters:
        ----------
        ac_space        action space

        policy_network  keras network for policy

        value_network   keras network for value

        estimate_q      q value or v value

        """

        self.policy_network = policy_network
        self.value_network = value_network or policy_network
        self.estimate_q = estimate_q
        self.initial_state = None

        # Based on the action space, will select what probability distribution type
        self.pdtype = make_pdtype(policy_network.output_shape, ac_space, init_scale=0.01)

        if estimate_q:
            assert isinstance(ac_space, gym.spaces.Discrete)
            self.value_fc = fc(self.value_network.output_shape, 'q', ac_space.n)
        else:
            self.value_fc = fc(self.value_network.output_shape, 'vf', 1)

    @tf.function
    def step(self, observation):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     batched observation data

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """

        latent = self.policy_network(observation)
        pd, pi = self.pdtype.pdfromlatent(latent)
        action = pd.sample()
        neglogp = pd.neglogp(action)
        value_latent = self.value_network(observation)
        vf = tf.squeeze(self.value_fc(value_latent), axis=1)
        return action, vf, None, neglogp

    @tf.function
    def value(self, observation):
        """
        Compute value estimate(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        Returns:
        -------
        value estimate
        """
        value_latent = self.value_network(observation)
        result = tf.squeeze(self.value_fc(value_latent), axis=1)
        return result

