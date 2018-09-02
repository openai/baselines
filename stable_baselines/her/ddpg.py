from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.contrib.staging import StagingArea

from stable_baselines import logger
from stable_baselines.her.util import import_function, flatten_grads, transitions_in_episode_batch
from stable_baselines.her.normalizer import Normalizer
from stable_baselines.her.replay_buffer import ReplayBuffer
from stable_baselines.common.mpi_adam import MpiAdam


def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}


class DDPG(object):
    def __init__(self, input_dims, buffer_size, hidden, layers, network_class, polyak, batch_size,
                 q_lr, pi_lr, norm_eps, norm_clip, max_u, action_l2, clip_obs, scope, time_horizon,
                 rollout_batch_size, subtract_goals, relative_goals, clip_pos_returns, clip_return,
                 sample_transitions, gamma, reuse=False):
        """
        Implementation of DDPG that is used in combination with Hindsight Experience Replay (HER).

        :param input_dims: ({str: int}) dimensions for the observation (o), the goal (g), and the actions (u)
        :param buffer_size: (int) number of transitions that are stored in the replay buffer
        :param hidden: (int) number of units in the hidden layers
        :param layers: (int) number of hidden layers
        :param network_class: (str) the network class that should be used (e.g. 'stable_baselines.her.ActorCritic')
        :param polyak: (float) coefficient for Polyak-averaging of the target network
        :param batch_size: (int) batch size for training
        :param q_lr: (float) learning rate for the Q (critic) network
        :param pi_lr: (float) learning rate for the pi (actor) network
        :param norm_eps: (float) a small value used in the normalizer to avoid numerical instabilities
        :param norm_clip: (float) normalized inputs are clipped to be in [-norm_clip, norm_clip]
        :param max_u: (float) maximum action magnitude, i.e. actions are in [-max_u, max_u]
        :param action_l2: (float) coefficient for L2 penalty on the actions
        :param clip_obs: (float) clip observations before normalization to be in [-clip_obs, clip_obs]
        :param scope: (str) the scope used for the TensorFlow graph
        :param time_horizon: (int) the time horizon for rollouts
        :param rollout_batch_size: (int) number of parallel rollouts per DDPG agent
        :param subtract_goals: (function (np.ndarray, np.ndarray): np.ndarray) function that subtracts goals
            from each other
        :param relative_goals: (boolean) whether or not relative goals should be fed into the network
        :param clip_pos_returns: (boolean) whether or not positive returns should be clipped
        :param clip_return: (float) clip returns to be in [-clip_return, clip_return]
        :param sample_transitions: (function (dict, int): dict) function that samples from the replay buffer
        :param gamma: (float) gamma used for Q learning updates
        :param reuse: (boolean) whether or not the networks should be reused
        """
        # Updated in experiments/config.py
        self.input_dims = input_dims
        self.buffer_size = buffer_size
        self.hidden = hidden
        self.layers = layers
        self.network_class = network_class
        self.polyak = polyak
        self.batch_size = batch_size
        self.q_lr = q_lr
        self.pi_lr = pi_lr
        self.norm_eps = norm_eps
        self.norm_clip = norm_clip
        self.max_u = max_u
        self.action_l2 = action_l2
        self.clip_obs = clip_obs
        self.scope = scope
        self.time_horizon = time_horizon
        self.rollout_batch_size = rollout_batch_size
        self.subtract_goals = subtract_goals
        self.relative_goals = relative_goals
        self.clip_pos_returns = clip_pos_returns
        self.clip_return = clip_return
        self.sample_transitions = sample_transitions
        self.gamma = gamma
        self.reuse = reuse

        if self.clip_return is None:
            self.clip_return = np.inf

        self.create_actor_critic = import_function(self.network_class)

        input_shapes = dims_to_shapes(self.input_dims)
        self.dim_obs = self.input_dims['o']
        self.dim_goal = self.input_dims['g']
        self.dim_action = self.input_dims['u']

        # Prepare staging area for feeding data to the model.
        stage_shapes = OrderedDict()
        for key in sorted(self.input_dims.keys()):
            if key.startswith('info_'):
                continue
            stage_shapes[key] = (None, *input_shapes[key])
        for key in ['o', 'g']:
            stage_shapes[key + '_2'] = stage_shapes[key]
        stage_shapes['r'] = (None,)
        self.stage_shapes = stage_shapes

        # Create network.
        with tf.variable_scope(self.scope):
            self.staging_tf = StagingArea(
                dtypes=[tf.float32 for _ in self.stage_shapes.keys()],
                shapes=list(self.stage_shapes.values()))
            self.buffer_ph_tf = [
                tf.placeholder(tf.float32, shape=shape) for shape in self.stage_shapes.values()]
            self.stage_op = self.staging_tf.put(self.buffer_ph_tf)

            self._create_network(reuse=reuse)

        # Configure the replay buffer.
        buffer_shapes = {key: (self.time_horizon if key != 'o' else self.time_horizon + 1, *input_shapes[key])
                         for key, val in input_shapes.items()}
        buffer_shapes['g'] = (buffer_shapes['g'][0], self.dim_goal)
        buffer_shapes['ag'] = (self.time_horizon + 1, self.dim_goal)

        buffer_size = (self.buffer_size // self.rollout_batch_size) * self.rollout_batch_size
        self.buffer = ReplayBuffer(buffer_shapes, buffer_size, self.time_horizon, self.sample_transitions)

    def _random_action(self, num):
        return np.random.uniform(low=-self.max_u, high=self.max_u, size=(num, self.dim_action))

    def _preprocess_obs_goal(self, obs, achieved_goal, goal):
        if self.relative_goals:
            g_shape = goal.shape
            goal = goal.reshape(-1, self.dim_goal)
            achieved_goal = achieved_goal.reshape(-1, self.dim_goal)
            goal = self.subtract_goals(goal, achieved_goal)
            goal = goal.reshape(*g_shape)
        obs = np.clip(obs, -self.clip_obs, self.clip_obs)
        goal = np.clip(goal, -self.clip_obs, self.clip_obs)
        return obs, goal

    def get_actions(self, obs, achieved_goal, goal, noise_eps=0., random_eps=0., use_target_net=False, compute_q=False):
        """
        return the action from an observation and goal

        :param obs: (np.ndarray) the observation
        :param achieved_goal: (np.ndarray) the achieved goal
        :param goal: (np.ndarray) the goal
        :param noise_eps: (float) the noise epsilon
        :param random_eps: (float) the random epsilon
        :param use_target_net: (bool) whether or not to use the target network
        :param compute_q: (bool) whether or not to compute Q value
        :return: (numpy float or float) the actions
        """
        obs, goal = self._preprocess_obs_goal(obs, achieved_goal, goal)
        policy = self.target if use_target_net else self.main
        # values to compute
        vals = [policy.pi_tf]
        if compute_q:
            vals += [policy.q_pi_tf]
        # feed
        feed = {
            policy.o_tf: obs.reshape(-1, self.dim_obs),
            policy.g_tf: goal.reshape(-1, self.dim_goal),
            policy.u_tf: np.zeros((obs.size // self.dim_obs, self.dim_action), dtype=np.float32)
        }

        ret = self.sess.run(vals, feed_dict=feed)
        # action postprocessing
        action = ret[0]
        noise = noise_eps * self.max_u * np.random.randn(*action.shape)  # gaussian noise
        action += noise
        action = np.clip(action, -self.max_u, self.max_u)
        # eps-greedy
        n_ac = action.shape[0]
        action += np.random.binomial(1, random_eps, n_ac).reshape(-1, 1) * (self._random_action(n_ac) - action)
        if action.shape[0] == 1:
            action = action[0]
        action = action.copy()
        ret[0] = action

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def store_episode(self, episode_batch, update_stats=True):
        """
        Story the episode transitions

        :param episode_batch: (np.ndarray) array of batch_size x (T or T+1) x dim_key 'o' is of size T+1,
            others are of size T
        :param update_stats: (bool) whether to update stats or not
        """

        self.buffer.store_episode(episode_batch)

        if update_stats:
            # add transitions to normalizer
            episode_batch['o_2'] = episode_batch['o'][:, 1:, :]
            episode_batch['ag_2'] = episode_batch['ag'][:, 1:, :]
            num_normalizing_transitions = transitions_in_episode_batch(episode_batch)
            transitions = self.sample_transitions(episode_batch, num_normalizing_transitions)

            obs, _, goal, achieved_goal = transitions['o'], transitions['o_2'], transitions['g'], transitions['ag']
            transitions['o'], transitions['g'] = self._preprocess_obs_goal(obs, achieved_goal, goal)
            # No need to preprocess the o_2 and g_2 since this is only used for stats

            self.o_stats.update(transitions['o'])
            self.g_stats.update(transitions['g'])

            self.o_stats.recompute_stats()
            self.g_stats.recompute_stats()

    def get_current_buffer_size(self):
        """
        returns the current buffer size

        :return: (int) buffer size
        """
        return self.buffer.get_current_size()

    def _sync_optimizers(self):
        self.q_adam.sync()
        self.pi_adam.sync()

    def _grads(self):
        # Avoid feed_dict here for performance!
        critic_loss, actor_loss, q_grad, pi_grad = self.sess.run([
            self.q_loss_tf,
            self.main.q_pi_tf,
            self.q_grad_tf,
            self.pi_grad_tf
        ])
        return critic_loss, actor_loss, q_grad, pi_grad

    def _update(self, q_grad, pi_grad):
        self.q_adam.update(q_grad, self.q_lr)
        self.pi_adam.update(pi_grad, self.pi_lr)

    def sample_batch(self):
        """
        sample a batch

        :return: (dict) the batch
        """
        transitions = self.buffer.sample(self.batch_size)
        obs, obs_2, goal = transitions['o'], transitions['o_2'], transitions['g']
        achieved_goal, achieved_goal_2 = transitions['ag'], transitions['ag_2']
        transitions['o'], transitions['g'] = self._preprocess_obs_goal(obs, achieved_goal, goal)
        transitions['o_2'], transitions['g_2'] = self._preprocess_obs_goal(obs_2, achieved_goal_2, goal)

        transitions_batch = [transitions[key] for key in self.stage_shapes.keys()]
        return transitions_batch

    def stage_batch(self, batch=None):
        """
        apply a batch to staging

        :param batch: (dict) the batch to add to staging, if None: self.sample_batch()
        """
        if batch is None:
            batch = self.sample_batch()
        assert len(self.buffer_ph_tf) == len(batch)
        self.sess.run(self.stage_op, feed_dict=dict(zip(self.buffer_ph_tf, batch)))

    def train(self, stage=True):
        """
        train DDPG

        :param stage: (bool) enable staging
        :return: (float, float) critic loss, actor loss
        """
        if stage:
            self.stage_batch()
        critic_loss, actor_loss, q_grad, pi_grad = self._grads()
        self._update(q_grad, pi_grad)
        return critic_loss, actor_loss

    def _init_target_net(self):
        self.sess.run(self.init_target_net_op)

    def update_target_net(self):
        """
        update the target network
        """
        self.sess.run(self.update_target_net_op)

    def clear_buffer(self):
        """
        clears the replay buffer
        """
        self.buffer.clear_buffer()

    def _vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + scope)
        assert len(res) > 0
        return res

    def _global_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + scope)
        return res

    def _create_network(self, reuse=False):
        logger.info("Creating a DDPG agent with action space %d x %s..." % (self.dim_action, self.max_u))

        self.sess = tf.get_default_session()
        if self.sess is None:
            self.sess = tf.InteractiveSession()

        # running averages
        with tf.variable_scope('o_stats') as scope:
            if reuse:
                scope.reuse_variables()
            self.o_stats = Normalizer(self.dim_obs, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.variable_scope('g_stats') as scope:
            if reuse:
                scope.reuse_variables()
            self.g_stats = Normalizer(self.dim_goal, self.norm_eps, self.norm_clip, sess=self.sess)

        # mini-batch sampling.
        batch = self.staging_tf.get()
        batch_tf = OrderedDict([(key, batch[i])
                                for i, key in enumerate(self.stage_shapes.keys())])
        batch_tf['r'] = tf.reshape(batch_tf['r'], [-1, 1])

        # networks
        with tf.variable_scope('main') as scope:
            if reuse:
                scope.reuse_variables()
            self.main = self.create_actor_critic(batch_tf, net_type='main', **self.__dict__)
            scope.reuse_variables()
        with tf.variable_scope('target') as scope:
            if reuse:
                scope.reuse_variables()
            target_batch_tf = batch_tf.copy()
            target_batch_tf['o'] = batch_tf['o_2']
            target_batch_tf['g'] = batch_tf['g_2']
            self.target = self.create_actor_critic(
                target_batch_tf, net_type='target', **self.__dict__)
            scope.reuse_variables()
        assert len(self._vars("main")) == len(self._vars("target"))

        # loss functions
        target_q_pi_tf = self.target.q_pi_tf
        clip_range = (-self.clip_return, 0. if self.clip_pos_returns else np.inf)
        target_tf = tf.clip_by_value(batch_tf['r'] + self.gamma * target_q_pi_tf, *clip_range)

        self.q_loss_tf = tf.reduce_mean(tf.square(tf.stop_gradient(target_tf) - self.main.q_tf))
        self.pi_loss_tf = -tf.reduce_mean(self.main.q_pi_tf)
        self.pi_loss_tf += self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))

        q_grads_tf = tf.gradients(self.q_loss_tf, self._vars('main/Q'))
        pi_grads_tf = tf.gradients(self.pi_loss_tf, self._vars('main/pi'))

        assert len(self._vars('main/Q')) == len(q_grads_tf)
        assert len(self._vars('main/pi')) == len(pi_grads_tf)

        self.q_grads_vars_tf = zip(q_grads_tf, self._vars('main/Q'))
        self.pi_grads_vars_tf = zip(pi_grads_tf, self._vars('main/pi'))
        self.q_grad_tf = flatten_grads(grads=q_grads_tf, var_list=self._vars('main/Q'))
        self.pi_grad_tf = flatten_grads(grads=pi_grads_tf, var_list=self._vars('main/pi'))

        # optimizers
        self.q_adam = MpiAdam(self._vars('main/Q'), scale_grad_by_procs=False)
        self.pi_adam = MpiAdam(self._vars('main/pi'), scale_grad_by_procs=False)

        # polyak averaging
        self.main_vars = self._vars('main/Q') + self._vars('main/pi')
        self.target_vars = self._vars('target/Q') + self._vars('target/pi')
        self.stats_vars = self._global_vars('o_stats') + self._global_vars('g_stats')
        self.init_target_net_op = list(
            map(lambda v: v[0].assign(v[1]), zip(self.target_vars, self.main_vars)))
        self.update_target_net_op = list(
            map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]),
                zip(self.target_vars, self.main_vars)))

        # initialize all variables
        tf.variables_initializer(self._global_vars('')).run()
        self._sync_optimizers()
        self._init_target_net()

    def logs(self, prefix=''):
        """
        create a log dictionary
        :param prefix: (str) the prefix for evey index
        :return: ({str: Any}) the log
        """
        logs = []
        logs += [('stats_o/mean', np.mean(self.sess.run([self.o_stats.mean])))]
        logs += [('stats_o/std', np.mean(self.sess.run([self.o_stats.std])))]
        logs += [('stats_g/mean', np.mean(self.sess.run([self.g_stats.mean])))]
        logs += [('stats_g/std', np.mean(self.sess.run([self.g_stats.std])))]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def __getstate__(self):
        """Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
        excluded_subnames = ['_tf', '_op', '_vars', '_adam', 'buffer', 'sess', '_stats',
                             'main', 'target', 'lock', 'env', 'sample_transitions',
                             'stage_shapes', 'create_actor_critic']

        state = {k: v for k, v in self.__dict__.items() if all([subname not in k for subname in excluded_subnames])}
        state['buffer_size'] = self.buffer_size
        state['tf'] = self.sess.run([x for x in self._global_vars('') if 'buffer' not in x.name])
        return state

    def __setstate__(self, state):
        if 'sample_transitions' not in state:
            # We don't need this for playing the policy.
            state['sample_transitions'] = None

        self.__init__(**state)
        # set up stats (they are overwritten in __init__)
        for key, value in state.items():
            if key[-6:] == '_stats':
                self.__dict__[key] = value
        # load TF variables
        _vars = [x for x in self._global_vars('') if 'buffer' not in x.name]
        assert len(_vars) == len(state["tf"])
        node = [tf.assign(var, val) for var, val in zip(_vars, state["tf"])]
        self.sess.run(node)
