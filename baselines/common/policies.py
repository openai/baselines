from collections import defaultdict

import tensorflow as tf
from baselines.common import tf_util
from baselines.a2c.utils import fc
from baselines.common.distributions import make_pdtype
from baselines.common.input import observation_placeholder, encode_observation
from baselines.common.tf_util import adjust_shape
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common.models import get_network_builder

import gym


class PolicyWithValue(object):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, env, observations, pol_latent, pol_head, estimate_q=False, vf_latent=None, vf_head=None, sess=None, **tensors):
        """
        Parameters:
        ----------
        env             RL environment

        observations    tensorflow placeholder in which the observations will be fed

        latent          latent state from which policy distribution parameters should be inferred

        vf_latent       latent state from which value function should be inferred (if None, then latent is used)

        sess            tensorflow session to run calculations in (if None, default session is used)

        **tensors       tensorflow tensors for additional attributes such as state or mask

        """

        self.X = observations
        self.state = tf.constant([])
        self.initial_state = None
        self.__dict__.update(tensors)

        # vf_latent = vf_latent if vf_latent is not None else latent

        # vf_latent = tf.layers.flatten(vf_latent)
        # latent = tf.layers.flatten(latent)

        # Based on the action space, will select what probability distribution type
        self.pdtype = make_pdtype(env.action_space)

        if pol_head is None:
            pol_head = tf.layers.flatten(pol_latent)
            # pol_head = _matching_fc(pol_latent, 'pi', self.pdtype.size, init_scale=0.01, init_bias=0.0)
        self.pd, self.pi = self.pdtype.pdfromlatent(pol_head, init_scale=0.01)

        # Take an action
        self.action = self.pd.sample()

        # Calculate the neg log of our probability
        self.neglogp = self.pd.neglogp(self.action)
        self.sess = sess or tf.get_default_session()

        if estimate_q:
            assert isinstance(env.action_space, gym.spaces.Discrete)
            if vf_head:
                if vf_head.size[-1] != env.action_space.n:
                    raise RuntimeError('When specifying the Q-function network head (instead of the latent representation), the size of the output must equal the no. of actions.')
                self.q = vf_head
            else:
                self.q = fc(vf_latent, 'q', env.action_space.n)
            self.vf = self.q
        else:
            if vf_head:
                if vf_head.size[-1] != 1:
                    raise RuntimeError('Value function must have a single output.')
                self.vf = vf_head
            else:
                self.vf = fc(vf_latent, 'vf', 1)
            self.vf = self.vf[:,0]

    def _evaluate(self, variables, observation, **extra_feed):
        sess = self.sess
        feed_dict = {self.X: adjust_shape(self.X, observation)}
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)

        return sess.run(variables, feed_dict)

    def step(self, observation, **extra_feed):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """

        a, v, state, neglogp = self._evaluate([self.action, self.vf, self.state, self.neglogp], observation, **extra_feed)
        if state.size == 0:
            state = None
        return a, v, state, neglogp

    def value(self, ob, *args, **kwargs):
        """
        Compute value estimate(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        value estimate
        """
        return self._evaluate(self.vf, ob, *args, **kwargs)

    def save(self, save_path):
        tf_util.save_state(save_path, sess=self.sess)

    def load(self, load_path):
        tf_util.load_state(load_path, sess=self.sess)

def build_policy(env, network_builder, value_network=None, normalize_observations=False, estimate_q=False, **policy_kwargs):
    is_recurrent = None # In case network_builder is a function, not str, in which case there is no way to know if it will return a recurrent network or not.
    if isinstance(network_builder, str):
        network_type = network_builder
        network_builder_dict = get_network_builder(network_type)
        network_builder = network_builder_dict['func'](**policy_kwargs)
        is_recurrent = network_builder_dict['is_recurrent']

    def policy_fn(nbatch=None, nsteps=None, sess=None, observ_placeholder=None):
        ob_space = env.observation_space

        X = observ_placeholder if observ_placeholder is not None else observation_placeholder(ob_space, batch_size=nbatch)

        extra_tensors = {}

        if normalize_observations and X.dtype == tf.float32:
            encoded_x, rms = _normalize_clip_observation(X)
            extra_tensors['rms'] = rms
        else:
            encoded_x = X

        encoded_x = encode_observation(ob_space, encoded_x)

        if is_recurrent:
            nenv = nbatch // nsteps
            assert nenv > 0, 'Bad input for recurrent policy: batch size {} smaller than nsteps {}'.format(nbatch, nsteps)
            policy = network_builder(encoded_x, 'pi', 'vf', value_network == 'copy', nenv=nenv)
        else:
            policy = network_builder(encoded_x, 'pi', 'vf', value_network == 'copy')
        if not isinstance(policy, dict):
            policy = {'latent': policy}
        policy = defaultdict(lambda: None, policy)

        if policy['recurrent_tensors'] is not None and is_recurrent is None:
            # recurrent architecture, need a few more steps
            nenv = nbatch // nsteps
            assert nenv > 0, 'Bad input for recurrent policy: batch size {} smaller than nsteps {}'.format(nbatch, nsteps)
            policy = network_builder(encoded_x, nenv)
            policy = defaultdict(lambda: None, policy)

        if policy['recurrent_tensors'] is not None:
            extra_tensors.update(policy['recurrent_tensors'])

        if policy['policy_latent'] is None:
            policy['policy_latent'] = policy['latent']

        # TODO recurrent architectures are not supported with value_network=copy yet
        if policy['value_latent'] is None:
            policy['value_latent'] = policy['latent']

        policy = PolicyWithValue(
            env=env,
            observations=X,
            pol_latent=policy['policy_latent'],
            pol_head=policy['policy_head'],
            vf_latent=policy['value_latent'],
            vf_head=policy['value_head'],
            sess=sess,
            estimate_q=estimate_q,
            **extra_tensors
        )
        return policy

    return policy_fn


def _normalize_clip_observation(x, clip_range=[-5.0, 5.0]):
    rms = RunningMeanStd(shape=x.shape[1:])
    norm_x = tf.clip_by_value((x - rms.mean) / rms.std, min(clip_range), max(clip_range))
    return norm_x, rms

