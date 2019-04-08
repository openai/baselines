import gym
import numpy as np
import tensorflow as tf

from baselines.a2c.utils import fc
from baselines.common import tf_util
from baselines.common.distributions import make_pdtype
from baselines.common.input import observation_placeholder, encode_observation
from baselines.common.models import RNN
from baselines.common.models import get_network_builder
from baselines.common.tf_util import adjust_shape


class PolicyWithValue(object):
    """
    Encapsulates fields and methods for RL policy and two value function estimation with shared parameters
    """

    def __init__(self, env, observations, latent, dones, states=None, estimate_q=False, vf_latent=None, sess=None):
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
        self.dones = dones
        self.pdtype = make_pdtype(env.action_space)
        self.states = states
        self.sess = sess or tf.get_default_session()

        vf_latent = vf_latent if vf_latent is not None else latent

        with tf.variable_scope('policy'):
            latent = tf.layers.flatten(latent)
            # Based on the action space, will select what probability distribution type
            self.pd, self.pi = self.pdtype.pdfromlatent(latent, init_scale=0.01)

            with tf.variable_scope('sample_action'):
                self.action = self.pd.sample()

            with tf.variable_scope('negative_log_probability'):
                # Calculate the neg log of our probability
                self.neglogp = self.pd.neglogp(self.action)

        with tf.variable_scope('value'):
            vf_latent = tf.layers.flatten(vf_latent)

            if estimate_q:
                assert isinstance(env.action_space, gym.spaces.Discrete)
                self.q = fc(vf_latent, 'q', env.action_space.n)
                self.value = self.q
            else:
                vf_latent = tf.layers.flatten(vf_latent)
                self.value = fc(vf_latent, 'value', 1, init_scale=0.01)
                self.value = self.value[:, 0]

        self.step_input = {
            'observations': observations,
            'dones': self.dones,
        }

        self.step_output = {
            'actions': self.action,
            'values': self.value,
            'neglogpacs': self.neglogp,
        }
        if self.states:
            self.initial_state = np.zeros(self.states['current'].get_shape())
            self.step_input.update({'states': self.states['current']})
            self.step_output.update({'states': self.states['current'],
                                     'next_states': self.states['next']})
        else:
            self.initial_state = None

    def feed_dict(self, **kwargs):
        feed_dict = {}
        for key in kwargs:
            if key in self.step_input:
                feed_dict[self.step_input[key]] = adjust_shape(self.step_input[key], kwargs[key])
        return feed_dict

    def step(self, **kwargs):
        return self.sess.run(self.step_output,
                             feed_dict=self.feed_dict(**kwargs))

    def values(self, **kwargs):
        return self.sess.run({'values': self.value},
                             feed_dict=self.feed_dict(**kwargs))

    def save(self, save_path):
        tf_util.save_state(save_path, sess=self.sess)

    def load(self, load_path):
        tf_util.load_state(load_path, sess=self.sess)


def build_ppo_policy(env, policy_network, value_network=None, estimate_q=False, **policy_kwargs):
    if isinstance(policy_network, str):
        network_type = policy_network
        policy_network = get_network_builder(network_type)(**policy_kwargs)

    if value_network is None:
        value_network = 'shared'

    def policy_fn(nbatch=None, nsteps=None, sess=None, observ_placeholder=None):
        next_states_list = []
        state_map = {}
        state_placeholder = None

        ob_space = env.observation_space
        X = observ_placeholder if observ_placeholder is not None else observation_placeholder(ob_space,
                                                                                              batch_size=nbatch)
        dones = tf.placeholder(tf.float32, shape=[X.shape[0]], name='dones')
        encoded_x = encode_observation(ob_space, X)

        with tf.variable_scope('current_rnn_memory'):
            if value_network == 'shared':
                value_network_ = value_network
            else:
                if value_network == 'copy':
                    value_network_ = policy_network
                else:
                    assert callable(value_network)
                    value_network_ = value_network

            policy_memory_size = policy_network.memory_size if isinstance(policy_network, RNN) else 0
            value_memory_size = value_network_.memory_size if isinstance(value_network_, RNN) else 0
            state_size = policy_memory_size + value_memory_size

            if state_size > 0:
                state_placeholder = tf.placeholder(dtype=tf.float32, shape=(nbatch, state_size),
                                                   name='states')

                state_map['policy'] = state_placeholder[:, 0:policy_memory_size]
                state_map['value'] = state_placeholder[:, policy_memory_size:]

        with tf.variable_scope('policy_latent', reuse=tf.AUTO_REUSE):
            if isinstance(policy_network, RNN):
                assert policy_memory_size > 0
                policy_latent, next_policy_state = \
                    policy_network(encoded_x, dones, state_map['policy'])
                next_states_list.append(next_policy_state)
            else:
                policy_latent = policy_network(encoded_x)

        with tf.variable_scope('value_latent', reuse=tf.AUTO_REUSE):
            if value_network_ == 'shared':
                value_latent = policy_latent
            elif isinstance(value_network_, RNN):
                assert value_memory_size > 0
                value_latent, next_value_state = \
                    value_network_(encoded_x, dones, state_map['value'])
                next_states_list.append(next_value_state)
            else:
                value_latent = value_network_(encoded_x)

        with tf.name_scope("next_rnn_memory"):
            if state_size > 0:
                next_states = tf.concat(next_states_list, axis=1)
                state_info = {'current': state_placeholder,
                              'next': next_states, }
            else:
                state_info = None

        policy = PolicyWithValue(
            env=env,
            observations=X,
            dones=dones,
            latent=policy_latent,
            vf_latent=value_latent,
            states=state_info,
            sess=sess,
            estimate_q=estimate_q,
        )
        return policy

    return policy_fn
