from functools import reduce
import os
import time
from collections import deque
import pickle

import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
from mpi4py import MPI

from stable_baselines import logger
from stable_baselines.common import tf_util, BaseRLModel, SetVerbosity
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.mpi_adam import MpiAdam
from stable_baselines.common.policies import LstmPolicy
from stable_baselines.common.mpi_running_mean_std import RunningMeanStd
from stable_baselines.a2c.utils import find_trainable_variables
from stable_baselines.ddpg.memory import Memory


def normalize(tensor, stats):
    """
    normalize a tensor using a running mean and std

    :param tensor: (TensorFlow Tensor) the input tensor
    :param stats: (RunningMeanStd) the running mean and std of the input to normalize
    :return: (TensorFlow Tensor) the normalized tensor
    """
    if stats is None:
        return tensor
    return (tensor - stats.mean) / stats.std


def denormalize(tensor, stats):
    """
    denormalize a tensor using a running mean and std

    :param tensor: (TensorFlow Tensor) the normalized tensor
    :param stats: (RunningMeanStd) the running mean and std of the input to normalize
    :return: (TensorFlow Tensor) the restored tensor
    """
    if stats is None:
        return tensor
    return tensor * stats.std + stats.mean


def reduce_std(tensor, axis=None, keepdims=False):
    """
    get the standard deviation of a Tensor

    :param tensor: (TensorFlow Tensor) the input tensor
    :param axis: (int or [int]) the axis to itterate the std over
    :param keepdims: (bool) keep the other dimensions the same
    :return: (TensorFlow Tensor) the std of the tensor
    """
    return tf.sqrt(reduce_var(tensor, axis=axis, keepdims=keepdims))


def reduce_var(tensor, axis=None, keepdims=False):
    """
    get the variance of a Tensor

    :param tensor: (TensorFlow Tensor) the input tensor
    :param axis: (int or [int]) the axis to itterate the variance over
    :param keepdims: (bool) keep the other dimensions the same
    :return: (TensorFlow Tensor) the variance of the tensor
    """
    tensor_mean = tf.reduce_mean(tensor, axis=axis, keepdims=True)
    devs_squared = tf.square(tensor - tensor_mean)
    return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims)


def get_target_updates(_vars, target_vars, tau):
    """
    get target update operations

    :param _vars: ([TensorFlow Tensor]) the initial variables
    :param target_vars: ([TensorFlow Tensor]) the target variables
    :param tau: (float) the soft update coefficient (keep old values, between 0 and 1)
    :return: (TensorFlow Operation, TensorFlow Operation) initial update, soft update
    """
    logger.info('setting up target updates ...')
    soft_updates = []
    init_updates = []
    assert len(_vars) == len(target_vars)
    for var, target_var in zip(_vars, target_vars):
        logger.info('  {} <- {}'.format(target_var.name, var.name))
        init_updates.append(tf.assign(target_var, var))
        soft_updates.append(tf.assign(target_var, (1. - tau) * target_var + tau * var))
    assert len(init_updates) == len(_vars)
    assert len(soft_updates) == len(_vars)
    return tf.group(*init_updates), tf.group(*soft_updates)


def get_perturbed_actor_updates(actor, perturbed_actor, param_noise_stddev):
    """
    get the actor update, with noise.

    :param actor: (str) the actor
    :param perturbed_actor: (str) the pertubed actor
    :param param_noise_stddev: (float) the std of the parameter noise
    :return: (TensorFlow Operation) the update function
    """
    assert len(tf_util.get_globals_vars(actor)) == len(tf_util.get_globals_vars(perturbed_actor))
    assert len([var for var in tf_util.get_trainable_vars(actor) if 'LayerNorm' not in var.name]) == \
        len([var for var in tf_util.get_trainable_vars(perturbed_actor) if 'LayerNorm' not in var.name])

    updates = []
    for var, perturbed_var in zip(tf_util.get_globals_vars(actor), tf_util.get_globals_vars(perturbed_actor)):
        if var in [var for var in tf_util.get_trainable_vars(actor) if 'LayerNorm' not in var.name]:
            logger.info('  {} <- {} + noise'.format(perturbed_var.name, var.name))
            updates.append(tf.assign(perturbed_var,
                                     var + tf.random_normal(tf.shape(var), mean=0., stddev=param_noise_stddev)))
        else:
            logger.info('  {} <- {}'.format(perturbed_var.name, var.name))
            updates.append(tf.assign(perturbed_var, var))
    assert len(updates) == len(tf_util.get_globals_vars(actor))
    return tf.group(*updates)


class DDPG(BaseRLModel):
    """
    Deep Deterministic Policy Gradient (DDPG) model

    DDPG: https://arxiv.org/pdf/1509.02971.pdf

    :param policy: (ActorCriticPolicy) the policy
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) the discount rate
    :param memory_policy: (Memory) the replay buffer (if None, default to baselines.ddpg.memory.Memory)
    :param eval_env: (Gym Environment) the evaluation environment (can be None)
    :param nb_train_steps: (int) the number of training steps
    :param nb_rollout_steps: (int) the number of rollout steps
    :param nb_eval_steps: (int) the number of evalutation steps
    :param param_noise: (AdaptiveParamNoiseSpec) the parameter noise type (can be None)
    :param action_noise: (ActionNoise) the action noise type (can be None)
    :param param_noise_adaption_interval: (int) apply param noise every N steps
    :param tau: (float) the soft update coefficient (keep old values, between 0 and 1)
    :param normalize_returns: (bool) should the critic output be normalized
    :param enable_popart: (bool) enable pop-art normalization of the critic output
        (https://arxiv.org/pdf/1602.07714.pdf)
    :param normalize_observations: (bool) should the observation be normalized
    :param batch_size: (int) the size of the batch for learning the policy
    :param observation_range: (tuple) the bounding values for the observation
    :param action_range: (tuple) the bounding values for the actions
    :param return_range: (tuple) the bounding values for the critic output
    :param critic_l2_reg: (float) l2 regularizer coefficient
    :param actor_lr: (float) the actor learning rate
    :param critic_lr: (float) the critic learning rate
    :param clip_norm: (float) clip the gradients (disabled if None)
    :param reward_scale: (float) the value the reward should be scaled by
    :param render: (bool) enable rendering of the environment
    :param render_eval: (bool) enable rendering of the evalution environment
    :param layer_norm: (bool) enable layer normalization for the policies
    :param memory_limit: (int) the max number of transitions to store
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    """

    def __init__(self, policy, env, gamma=0.99, memory_policy=None, eval_env=None,
                 nb_train_steps=50, nb_rollout_steps=100, nb_eval_steps=100, param_noise=None, action_noise=None,
                 action_range=(-1., 1.), normalize_observations=False, tau=0.001, batch_size=128,
                 param_noise_adaption_interval=50, normalize_returns=False, enable_popart=False,
                 observation_range=(-5., 5.), critic_l2_reg=0., return_range=(-np.inf, np.inf), actor_lr=1e-4,
                 critic_lr=1e-3, clip_norm=None, reward_scale=1., render=False, render_eval=False, layer_norm=True,
                 memory_limit=100, verbose=0, _init_setup_model=True):
        super(DDPG, self).__init__(policy=policy, env=env, requires_vec_env=False, verbose=verbose)

        # Parameters.
        self.gamma = gamma
        self.tau = tau
        self.memory_policy = memory_policy or Memory
        self.normalize_observations = normalize_observations
        self.normalize_returns = normalize_returns
        self.action_noise = action_noise
        self.param_noise = param_noise
        self.action_range = action_range
        self.return_range = return_range
        self.observation_range = observation_range
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.clip_norm = clip_norm
        self.enable_popart = enable_popart
        self.reward_scale = reward_scale
        self.batch_size = batch_size
        self.critic_l2_reg = critic_l2_reg
        self.eval_env = eval_env
        self.render = render
        self.render_eval = render_eval
        self.nb_eval_steps = nb_eval_steps
        self.param_noise_adaption_interval = param_noise_adaption_interval
        self.nb_train_steps = nb_train_steps
        self.nb_rollout_steps = nb_rollout_steps
        self.layer_norm = layer_norm
        self.memory_limit = memory_limit

        # init
        self.graph = None
        self.stats_sample = None
        self.memory = None
        self.policy_tf = None
        self.target_init_updates = None
        self.target_soft_updates = None
        self.critic_loss = None
        self.critic_grads = None
        self.critic_optimizer = None
        self.sess = None
        self.stats_ops = None
        self.stats_names = None
        self.perturbed_actor_tf = None
        self.perturb_policy_ops = None
        self.perturb_adaptive_policy_ops = None
        self.adaptive_policy_distance = None
        self.actor_loss = None
        self.actor_grads = None
        self.actor_optimizer = None
        self.old_std = None
        self.old_mean = None
        self.renormalize_q_outputs_op = None
        self.obs_rms = None
        self.ret_rms = None
        self.target_policy = None
        self.actor_tf = None
        self.normalized_critic_tf = None
        self.critic_tf = None
        self.normalized_critic_with_actor_tf = None
        self.critic_with_actor_tf = None
        self.target_q = None
        self.obs_train = None
        self.obs_target = None
        self.obs_noise = None
        self.obs_adapt_noise = None
        self.terminals1 = None
        self.rewards = None
        self.actions = None
        self.critic_target = None
        self.param_noise_stddev = None
        self.params = None

        if _init_setup_model:
            self.setup_model()

    def setup_model(self):
        with SetVerbosity(self.verbose):

            assert isinstance(self.action_space, gym.spaces.Box), \
                "Error: DDPG cannot output a {} action space, only spaces.Box is supported.".format(self.action_space)
            assert not issubclass(self.policy, LstmPolicy), "Error: cannot use a reccurent policy for the DDPG model."

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.sess = tf_util.single_threaded_session(graph=self.graph)

                self.memory = self.memory_policy(limit=self.memory_limit, action_shape=self.action_space.shape,
                                                 observation_shape=self.observation_space.shape)

                with tf.variable_scope("train", reuse=False):
                    self.policy_tf = self.policy(self.sess, self.observation_space, self.action_space, 1, 1, None)

                # Inputs.
                self.obs_train = self.policy_tf.obs_ph
                self.terminals1 = tf.placeholder(tf.float32, shape=(None, 1), name='terminals1')
                self.rewards = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
                self.actions = tf.placeholder(tf.float32, shape=(None,) + self.action_space.shape, name='actions')
                self.critic_target = tf.placeholder(tf.float32, shape=(None, 1), name='critic_target')
                self.param_noise_stddev = tf.placeholder(tf.float32, shape=(), name='param_noise_stddev')

                # Observation normalization.
                if self.normalize_observations:
                    with tf.variable_scope('obs_rms'):
                        self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
                else:
                    self.obs_rms = None

                # Return normalization.
                if self.normalize_returns:
                    with tf.variable_scope('ret_rms'):
                        self.ret_rms = RunningMeanStd()
                else:
                    self.ret_rms = None

                # Create target networks.
                with tf.variable_scope("target", reuse=False):
                    self.target_policy = self.policy(self.sess, self.observation_space, self.action_space, 1, 1, None)
                    self.obs_target = self.target_policy.obs_ph

                # Create networks and core TF parts that are shared across setup parts.
                self.actor_tf = self.policy_tf.policy
                self.normalized_critic_tf = self.policy_tf.value_fn
                self.critic_tf = denormalize(
                    tf.clip_by_value(self.normalized_critic_tf, self.return_range[0], self.return_range[1]),
                    self.ret_rms)
                self.normalized_critic_with_actor_tf = self.policy_tf.value_fn
                self.critic_with_actor_tf = denormalize(
                    tf.clip_by_value(self.normalized_critic_with_actor_tf, self.return_range[0], self.return_range[1]),
                    self.ret_rms)
                q_obs1 = denormalize(self.target_policy.value_fn, self.ret_rms)
                self.target_q = self.rewards + (1. - self.terminals1) * self.gamma * q_obs1

                # Set up parts.
                if self.param_noise is not None:
                    self._setup_param_noise()
                self._setup_actor_optimizer()
                self._setup_critic_optimizer()
                if self.normalize_returns and self.enable_popart:
                    self._setup_popart()
                self._setup_stats()
                self._setup_target_network_updates()

                self.params = find_trainable_variables("train")

                with self.sess.as_default():
                    self._initialize(self.sess)

    def _setup_target_network_updates(self):
        """
        set the target update operations
        """
        init_updates, soft_updates = get_target_updates(tf_util.get_trainable_vars('train'),
                                                        tf_util.get_trainable_vars('target'), self.tau)
        self.target_init_updates = init_updates
        self.target_soft_updates = soft_updates

    def _setup_param_noise(self):
        """
        set the parameter noise operations
        """
        assert self.param_noise is not None

        # Configure perturbed actor.
        with tf.variable_scope("noise", reuse=False):
            param_noise_actor = self.policy(self.sess, self.observation_space, self.action_space, 1, 1, None)
            self.obs_noise = param_noise_actor.obs_ph
        self.perturbed_actor_tf = param_noise_actor.policy
        logger.info('setting up param noise')
        self.perturb_policy_ops = get_perturbed_actor_updates('train', 'noise', self.param_noise_stddev)

        # Configure separate copy for stddev adoption.
        with tf.variable_scope("noise_adapt", reuse=False):
            adaptive_param_noise_actor = self.policy(self.sess, self.observation_space, self.action_space, 1, 1, None)
            self.obs_adapt_noise = adaptive_param_noise_actor.obs_ph
        adaptive_actor_tf = adaptive_param_noise_actor.policy
        self.perturb_adaptive_policy_ops = get_perturbed_actor_updates('train', 'noise_adapt', self.param_noise_stddev)
        self.adaptive_policy_distance = tf.sqrt(tf.reduce_mean(tf.square(self.actor_tf - adaptive_actor_tf)))

    def _setup_actor_optimizer(self):
        """
        setup the optimizer for the actor
        """
        logger.info('setting up actor optimizer')
        self.actor_loss = -tf.reduce_mean(self.critic_with_actor_tf)
        actor_shapes = [var.get_shape().as_list() for var in tf_util.get_trainable_vars('train')]
        actor_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in actor_shapes])
        logger.info('  actor shapes: {}'.format(actor_shapes))
        logger.info('  actor params: {}'.format(actor_nb_params))
        self.actor_grads = tf_util.flatgrad(self.actor_loss, tf_util.get_trainable_vars('train'),
                                            clip_norm=self.clip_norm)
        self.actor_optimizer = MpiAdam(var_list=tf_util.get_trainable_vars('train'), beta1=0.9, beta2=0.999,
                                       epsilon=1e-08)

    def _setup_critic_optimizer(self):
        """
        setup the optimizer for the critic
        """
        logger.info('setting up critic optimizer')
        normalized_critic_target_tf = tf.clip_by_value(normalize(self.critic_target, self.ret_rms),
                                                       self.return_range[0], self.return_range[1])
        self.critic_loss = tf.reduce_mean(tf.square(self.normalized_critic_tf - normalized_critic_target_tf))
        if self.critic_l2_reg > 0.:
            critic_reg_vars = [var for var in tf_util.get_trainable_vars('train')
                               if 'bias' not in var.name and 'output' not in var.name and 'b' not in var.name]
            for var in critic_reg_vars:
                logger.info('  regularizing: {}'.format(var.name))
            logger.info('  applying l2 regularization with {}'.format(self.critic_l2_reg))
            critic_reg = tc.layers.apply_regularization(
                tc.layers.l2_regularizer(self.critic_l2_reg),
                weights_list=critic_reg_vars
            )
            self.critic_loss += critic_reg
        critic_shapes = [var.get_shape().as_list() for var in tf_util.get_trainable_vars('train')]
        critic_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in critic_shapes])
        logger.info('  critic shapes: {}'.format(critic_shapes))
        logger.info('  critic params: {}'.format(critic_nb_params))
        self.critic_grads = tf_util.flatgrad(self.critic_loss, tf_util.get_trainable_vars('train'),
                                             clip_norm=self.clip_norm)
        self.critic_optimizer = MpiAdam(var_list=tf_util.get_trainable_vars('train'), beta1=0.9, beta2=0.999,
                                        epsilon=1e-08)

    def _setup_popart(self):
        """
        setup pop-art normalization of the critic output

        See https://arxiv.org/pdf/1602.07714.pdf for details.
        Preserving Outputs Precisely, while Adaptively Rescaling Targets”.
        """
        self.old_std = tf.placeholder(tf.float32, shape=[1], name='old_std')
        new_std = self.ret_rms.std
        self.old_mean = tf.placeholder(tf.float32, shape=[1], name='old_mean')
        new_mean = self.ret_rms.mean

        self.renormalize_q_outputs_op = []
        for out_vars in [[var for var in tf_util.get_trainable_vars('train') if 'output' in var.name],
                         [var for var in tf_util.get_trainable_vars('target') if 'output' in var.name]]:
            assert len(out_vars) == 2
            # wieght and bias of the last layer
            weight, bias = out_vars
            assert 'kernel' in weight.name
            assert 'bias' in bias.name
            assert weight.get_shape()[-1] == 1
            assert bias.get_shape()[-1] == 1
            self.renormalize_q_outputs_op += [weight.assign(weight * self.old_std / new_std)]
            self.renormalize_q_outputs_op += [bias.assign((bias * self.old_std + self.old_mean - new_mean) / new_std)]

    def _setup_stats(self):
        """
        setup the running means and std of the inputs and outputs of the model
        """
        ops = []
        names = []

        if self.normalize_returns:
            ops += [self.ret_rms.mean, self.ret_rms.std]
            names += ['ret_rms_mean', 'ret_rms_std']

        if self.normalize_observations:
            ops += [tf.reduce_mean(self.obs_rms.mean), tf.reduce_mean(self.obs_rms.std)]
            names += ['obs_rms_mean', 'obs_rms_std']

        ops += [tf.reduce_mean(self.critic_tf)]
        names += ['reference_Q_mean']
        ops += [reduce_std(self.critic_tf)]
        names += ['reference_Q_std']

        ops += [tf.reduce_mean(self.critic_with_actor_tf)]
        names += ['reference_actor_Q_mean']
        ops += [reduce_std(self.critic_with_actor_tf)]
        names += ['reference_actor_Q_std']

        ops += [tf.reduce_mean(self.actor_tf)]
        names += ['reference_action_mean']
        ops += [reduce_std(self.actor_tf)]
        names += ['reference_action_std']

        if self.param_noise:
            ops += [tf.reduce_mean(self.perturbed_actor_tf)]
            names += ['reference_perturbed_action_mean']
            ops += [reduce_std(self.perturbed_actor_tf)]
            names += ['reference_perturbed_action_std']

        self.stats_ops = ops
        self.stats_names = names

    def _policy(self, obs, apply_noise=True, compute_q=True):
        """
        Get the actions and critic output, from a given observation

        :param obs: ([float] or [int]) the observation
        :param apply_noise: (bool) enable the noise
        :param compute_q: (bool) compute the critic output
        :return: ([float], float) the action and critic value
        """
        feed_dict = {self.obs_train: [obs]}
        if self.param_noise is not None and apply_noise:
            actor_tf = self.perturbed_actor_tf
            feed_dict[self.obs_noise] = [obs]
        else:
            actor_tf = self.actor_tf
        if compute_q:
            action, q_value = self.sess.run([actor_tf, self.critic_with_actor_tf], feed_dict=feed_dict)
        else:
            action = self.sess.run(actor_tf, feed_dict=feed_dict)
            q_value = None
        action = action.flatten()
        if self.action_noise is not None and apply_noise:
            noise = self.action_noise()
            assert noise.shape == action.shape
            action += noise
        action = np.clip(action, self.action_range[0], self.action_range[1])
        return action, q_value

    def _store_transition(self, obs0, action, reward, obs1, terminal1):
        """
        Store a transition in the replay buffer

        :param obs0: ([float] or [int]) the last observation
        :param action: ([float]) the action
        :param reward: (float] the reward
        :param obs1: ([float] or [int]) the current observation
        :param terminal1: (bool) is the episode done
        """
        reward *= self.reward_scale
        self.memory.append(obs0, action, reward, obs1, terminal1)
        if self.normalize_observations:
            self.obs_rms.update(np.array([obs0]))

    def _train_step(self):
        """
        run a step of training from batch

        :return: (float, float) critic loss, actor loss
        """
        # Get a batch.
        batch = self.memory.sample(batch_size=self.batch_size)

        if self.normalize_returns and self.enable_popart:
            old_mean, old_std, target_q = self.sess.run([self.ret_rms.mean, self.ret_rms.std, self.target_q],
                                                        feed_dict={
                                                            self.obs_target: batch['obs1'],
                                                            self.rewards: batch['rewards'],
                                                            self.terminals1: batch['terminals1'].astype('float32'),
                                                        })
            self.ret_rms.update(target_q.flatten())
            self.sess.run(self.renormalize_q_outputs_op, feed_dict={
                self.old_std: np.array([old_std]),
                self.old_mean: np.array([old_mean]),
            })

        else:
            target_q = self.sess.run(self.target_q, feed_dict={
                self.obs_target: batch['obs1'],
                self.rewards: batch['rewards'],
                self.terminals1: batch['terminals1'].astype('float32'),
            })

        # Get all gradients and perform a synced update.
        ops = [self.actor_grads, self.actor_loss, self.critic_grads, self.critic_loss]
        actor_grads, actor_loss, critic_grads, critic_loss = self.sess.run(ops, feed_dict={
            self.obs_train: batch['obs0'],
            self.actions: batch['actions'],
            self.critic_target: target_q,
        })
        self.actor_optimizer.update(actor_grads, learning_rate=self.actor_lr)
        self.critic_optimizer.update(critic_grads, learning_rate=self.critic_lr)

        return critic_loss, actor_loss

    def _initialize(self, sess):
        """
        initialize the model parameters and optimizers

        :param sess: (TensorFlow Session) the current TensorFlow session
        """
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.actor_optimizer.sync()
        self.critic_optimizer.sync()
        self.sess.run(self.target_init_updates)

    def _update_target_net(self):
        """
        run target soft update operation
        """
        self.sess.run(self.target_soft_updates)

    def _get_stats(self):
        """
        Get the mean and standard deviation of the model's inputs and outputs

        :return: (dict) the means and stds
        """
        if self.stats_sample is None:
            # Get a sample and keep that fixed for all further computations.
            # This allows us to estimate the change in value for the same set of inputs.
            self.stats_sample = self.memory.sample(batch_size=self.batch_size)

        feed_dict = {
            self.actions: self.stats_sample['actions']
        }

        for placeholder in [self.obs_train, self.obs_target, self.obs_adapt_noise, self.obs_noise]:
            if placeholder is not None:
                feed_dict[placeholder] = self.stats_sample['obs0']

        values = self.sess.run(self.stats_ops, feed_dict=feed_dict)

        names = self.stats_names[:]
        assert len(names) == len(values)
        stats = dict(zip(names, values))

        if self.param_noise is not None:
            stats = {**stats, **self.param_noise.get_stats()}

        return stats

    def _adapt_param_noise(self):
        """
        calculate the adaptation for the parameter noise

        :return: (float) the mean distance for the parameter noise
        """
        if self.param_noise is None:
            return 0.

        # Perturb a separate copy of the policy to adjust the scale for the next "real" perturbation.
        batch = self.memory.sample(batch_size=self.batch_size)
        self.sess.run(self.perturb_adaptive_policy_ops, feed_dict={
            self.param_noise_stddev: self.param_noise.current_stddev,
        })
        distance = self.sess.run(self.adaptive_policy_distance, feed_dict={
            self.obs_adapt_noise: batch['obs0'], self.obs_train: batch['obs0'],
            self.param_noise_stddev: self.param_noise.current_stddev,
        })

        mean_distance = MPI.COMM_WORLD.allreduce(distance, op=MPI.SUM) / MPI.COMM_WORLD.Get_size()
        self.param_noise.adapt(mean_distance)
        return mean_distance

    def _reset(self):
        """
        Reset internal state after an episode is complete.
        """
        if self.action_noise is not None:
            self.action_noise.reset()
        if self.param_noise is not None:
            self.sess.run(self.perturb_policy_ops, feed_dict={
                self.param_noise_stddev: self.param_noise.current_stddev,
            })

    def learn(self, total_timesteps, callback=None, seed=None, log_interval=100):
        with SetVerbosity(self.verbose):
            self._setup_learn(seed)

            rank = MPI.COMM_WORLD.Get_rank()
            # we assume symmetric actions.
            assert np.all(np.abs(self.env.action_space.low) == self.env.action_space.high)
            max_action = self.env.action_space.high
            logger.log('scaling actions by {} before executing in env'.format(max_action))
            logger.log('Using agent with the following configuration:')
            logger.log(str(self.__dict__.items()))

            eval_episode_rewards_history = deque(maxlen=100)
            episode_rewards_history = deque(maxlen=100)
            with self.sess.as_default(), self.graph.as_default():
                # Prepare everything.
                self._reset()
                obs = self.env.reset()
                eval_obs = None
                if self.eval_env is not None:
                    eval_obs = self.eval_env.reset()
                episode_reward = 0.
                episode_step = 0
                episodes = 0
                step = 0
                total_steps = 0

                start_time = time.time()

                epoch_episode_rewards = []
                epoch_episode_steps = []
                epoch_actor_losses = []
                epoch_critic_losses = []
                epoch_adaptive_distances = []
                eval_episode_rewards = []
                eval_qs = []
                epoch_actions = []
                epoch_qs = []
                epoch_episodes = 0
                epoch = 0
                while True:
                    for _ in range(log_interval):
                        # Perform rollouts.
                        for _ in range(self.nb_rollout_steps):
                            if total_steps >= total_timesteps:
                                return self

                            # Predict next action.
                            action, q_value = self._policy(obs, apply_noise=True, compute_q=True)
                            assert action.shape == self.env.action_space.shape

                            # Execute next action.
                            if rank == 0 and self.render:
                                self.env.render()
                            assert max_action.shape == action.shape
                            # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                            new_obs, reward, done, _ = self.env.step(max_action * action)
                            step += 1
                            total_steps += 1
                            if rank == 0 and self.render:
                                self.env.render()
                            episode_reward += reward
                            episode_step += 1

                            # Book-keeping.
                            epoch_actions.append(action)
                            epoch_qs.append(q_value)
                            self._store_transition(obs, action, reward, new_obs, done)
                            obs = new_obs
                            if callback is not None:
                                callback(locals(), globals())

                            if done:
                                # Episode done.
                                epoch_episode_rewards.append(episode_reward)
                                episode_rewards_history.append(episode_reward)
                                epoch_episode_steps.append(episode_step)
                                episode_reward = 0.
                                episode_step = 0
                                epoch_episodes += 1
                                episodes += 1

                                self._reset()
                                if not isinstance(self.env, VecEnv):
                                    obs = self.env.reset()

                        # Train.
                        epoch_actor_losses = []
                        epoch_critic_losses = []
                        epoch_adaptive_distances = []
                        for t_train in range(self.nb_train_steps):
                            # Adapt param noise, if necessary.
                            if self.memory.nb_entries >= self.batch_size and \
                                    t_train % self.param_noise_adaption_interval == 0:
                                distance = self._adapt_param_noise()
                                epoch_adaptive_distances.append(distance)

                            critic_loss, actor_loss = self._train_step()
                            epoch_critic_losses.append(critic_loss)
                            epoch_actor_losses.append(actor_loss)
                            self._update_target_net()

                        # Evaluate.
                        eval_episode_rewards = []
                        eval_qs = []
                        if self.eval_env is not None:
                            eval_episode_reward = 0.
                            for _ in range(self.nb_eval_steps):
                                if total_steps >= total_timesteps:
                                    return self

                                eval_action, eval_q = self._policy(eval_obs, apply_noise=False, compute_q=True)
                                # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                                eval_obs, eval_r, eval_done, _ = self.eval_env.step(max_action * eval_action)
                                if self.render_eval:
                                    self.eval_env.render()
                                eval_episode_reward += eval_r

                                eval_qs.append(eval_q)
                                if eval_done:
                                    if not isinstance(self.env, VecEnv):
                                        eval_obs = self.eval_env.reset()
                                    eval_episode_rewards.append(eval_episode_reward)
                                    eval_episode_rewards_history.append(eval_episode_reward)
                                    eval_episode_reward = 0.

                    mpi_size = MPI.COMM_WORLD.Get_size()
                    # Log stats.
                    # XXX shouldn't call np.mean on variable length lists
                    duration = time.time() - start_time
                    stats = self._get_stats()
                    combined_stats = stats.copy()
                    combined_stats['rollout/return'] = np.mean(epoch_episode_rewards)
                    combined_stats['rollout/return_history'] = np.mean(episode_rewards_history)
                    combined_stats['rollout/episode_steps'] = np.mean(epoch_episode_steps)
                    combined_stats['rollout/actions_mean'] = np.mean(epoch_actions)
                    combined_stats['rollout/Q_mean'] = np.mean(epoch_qs)
                    combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
                    combined_stats['train/loss_critic'] = np.mean(epoch_critic_losses)
                    if len(epoch_adaptive_distances) != 0:
                        combined_stats['train/param_noise_distance'] = np.mean(epoch_adaptive_distances)
                    combined_stats['total/duration'] = duration
                    combined_stats['total/steps_per_second'] = float(step) / float(duration)
                    combined_stats['total/episodes'] = episodes
                    combined_stats['rollout/episodes'] = epoch_episodes
                    combined_stats['rollout/actions_std'] = np.std(epoch_actions)
                    # Evaluation statistics.
                    if self.eval_env is not None:
                        combined_stats['eval/return'] = eval_episode_rewards
                        combined_stats['eval/return_history'] = np.mean(eval_episode_rewards_history)
                        combined_stats['eval/Q'] = eval_qs
                        combined_stats['eval/episodes'] = len(eval_episode_rewards)

                    def as_scalar(scalar):
                        """
                        check and return the input if it is a scalar, otherwise raise ValueError

                        :param scalar: (Any) the object to check
                        :return: (Number) the scalar if x is a scalar
                        """
                        if isinstance(scalar, np.ndarray):
                            assert scalar.size == 1
                            return scalar[0]
                        elif np.isscalar(scalar):
                            return scalar
                        else:
                            raise ValueError('expected scalar, got %s' % scalar)

                    combined_stats_sums = MPI.COMM_WORLD.allreduce(
                        np.array([as_scalar(x) for x in combined_stats.values()]))
                    combined_stats = {k: v / mpi_size for (k, v) in zip(combined_stats.keys(), combined_stats_sums)}

                    # Total statistics.
                    combined_stats['total/epochs'] = epoch + 1
                    combined_stats['total/steps'] = step

                    for key in sorted(combined_stats.keys()):
                        logger.record_tabular(key, combined_stats[key])
                    logger.dump_tabular()
                    logger.info('')
                    logdir = logger.get_dir()
                    if rank == 0 and logdir:
                        if hasattr(self.env, 'get_state'):
                            with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as file_handler:
                                pickle.dump(self.env.get_state(), file_handler)
                        if self.eval_env and hasattr(self.eval_env, 'get_state'):
                            with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as file_handler:
                                pickle.dump(self.eval_env.get_state(), file_handler)

    def predict(self, observation, state=None, mask=None):
        observation = np.array(observation).reshape(self.observation_space.shape)

        action, _ = self._policy(observation, apply_noise=False, compute_q=True)
        if self._vectorize_action:
            return [action], [None]
        else:
            return action, None

    def action_probability(self, observation, state=None, mask=None):
        # here there are no action probabilities, as DDPG is continuous
        if self._vectorize_action:
            return self.sess.run(self.policy_tf.policy_proba, feed_dict={self.obs_train: observation})
        else:
            return self.sess.run(self.policy_tf.policy_proba, feed_dict={self.obs_train: observation})[0]

    def save(self, save_path):
        data = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "nb_eval_steps": self.nb_eval_steps,
            "param_noise_adaption_interval": self.param_noise_adaption_interval,
            "nb_train_steps": self.nb_train_steps,
            "nb_rollout_steps": self.nb_rollout_steps,
            "verbose": self.verbose,
            "param_noise": self.param_noise,
            "action_noise": self.action_noise,
            "gamma": self.gamma,
            "tau": self.tau,
            "normalize_returns": self.normalize_returns,
            "enable_popart": self.enable_popart,
            "normalize_observations": self.normalize_observations,
            "batch_size": self.batch_size,
            "observation_range": self.observation_range,
            "action_range": self.action_range,
            "return_range": self.return_range,
            "critic_l2_reg": self.critic_l2_reg,
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
            "clip_norm": self.clip_norm,
            "reward_scale": self.reward_scale,
            "layer_norm": self.layer_norm,
            "memory_limit": self.memory_limit,
            "policy": self.policy,
            "memory_policy": self.memory_policy,
            "n_envs": self.n_envs,
            "_vectorize_action": self._vectorize_action
        }

        params = self.sess.run(self.params)

        self._save_to_file(save_path, data=data, params=params)

    @classmethod
    def load(cls, load_path, env=None, **kwargs):
        data, params = cls._load_from_file(load_path)

        model = cls(None, env, _init_setup_model=False)
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model.set_env(env)
        model.setup_model()

        restores = []
        for param, loaded_p in zip(model.params, params):
            restores.append(param.assign(loaded_p))
        model.sess.run(restores)

        return model
