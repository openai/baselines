import tensorflow as tf

from stable_baselines.common.input import observation_input
from stable_baselines.common.distributions import make_proba_dist_type


class BasePolicy(object):
    def __init__(self, placeholders=None):
        """
        A base policy object for PPO1

        :param placeholders: (dict) To feed existing placeholders if needed
        """
        super(BasePolicy, self).__init__()
        self.sess = None
        self.pdtype = None
        self._act = None
        self.scope = None
        self.obs_ph = None
        self.stochastic_ph = None
        self.processed_x = None

        if placeholders is not None:
            self.obs_ph = placeholders.get("obs", None)
            self.processed_x = placeholders.get("processed_obs", None)
            self.stochastic_ph = placeholders.get("stochastic", None)

    def get_obs_and_pdtype(self, ob_space, ac_space):
        """
        Initialize probability distribution and get observation placeholder.

        :param ob_space: (Gym Spaces) the observation space
        :param ac_space: (Gym Spaces) the action space
        """
        self.pdtype = pdtype = make_proba_dist_type(ac_space)

        if self.obs_ph is None:
            self.obs_ph, self.processed_x = observation_input(ob_space)
        else:
            assert self.processed_x is not None

        return self.obs_ph, pdtype

    def act(self, stochastic, obs):
        """
        Get the action from the policy, using the observation

        :param stochastic: (bool) whether or not to use a stochastic or deterministic policy
        :param obs: (TensorFlow Tensor or np.ndarray) the observation
        :return: (np.ndarray, np.ndarray) the action and value function
        """
        ac1, vpred1 = self._act(stochastic, obs[None], sess=self.sess)
        return ac1[0], vpred1[0]

    def get_variables(self):
        """
        Get all the policy's variables

        :return: ([TensorFlow Tensor]) the variables of the network
        """
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        """
        Get the policy's trainable variables

        :return: ([TensorFlow Tensor]) the trainable variables of the network
        """
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    @classmethod
    def get_initial_state(cls):
        """
        Get the initial state

        :return: ([np.ndarray]) the initial state
        """
        return []
