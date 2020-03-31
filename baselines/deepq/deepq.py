import os
import pickle
import random
import tempfile
import gym
import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np
import copy
import time

import baselines.common.tf_util as U
from baselines.common.tf_util import load_variables, save_variables
from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines.common import set_global_seeds

from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, \
    DoublePrioritizedStateRecycledReplayBuffer
from baselines.deepq.utils import ObservationInput

from baselines.common.tf_util import get_session
from baselines.deepq.models import build_q_func


class ActWrapper(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params
        self.initial_state = None

    @staticmethod
    def load_act(path):
        with open(path, "rb") as f:
            model_data, act_params = cloudpickle.load(f)
        act = deepq.build_act(**act_params)
        sess = tf.Session()
        sess.__enter__()
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            load_variables(os.path.join(td, "model"))

        return ActWrapper(act, act_params)

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def step(self, observation, **kwargs):
        # DQN doesn't use RNNs so we ignore states and masks
        kwargs.pop('S', None)
        kwargs.pop('M', None)
        return self._act([observation], **kwargs), None, None, None

    def save_act(self, path=None):
        """Save model to a pickle located at `path`"""
        if path is None:
            path = os.path.join(logger.get_dir(), "model.pkl")

        with tempfile.TemporaryDirectory() as td:
            save_variables(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            cloudpickle.dump((model_data, self._act_params), f)

    def save(self, path):
        save_variables(path)


def load_act(path):
    """Load act function that was returned by learn function.

    Parameters
    ----------
    path: str
        path to the act function pickle

    Returns
    -------
    act: ActWrapper
        function that takes a batch of observations
        and returns actions.
    """
    return ActWrapper.load_act(path)


def learn(env,
          network,
          seed=None,
          lr=5e-4,
          total_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=100,
          checkpoint_freq=10000,
          checkpoint_path=None,
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          debug_flag=False,
          dpsr_replay=False,
          dpsr_replay_alpha1=0.6,
          dpsr_replay_alpha2=0.6,
          dpsr_replay_candidates_size=5,
          dpsr_common_replacement_candidates_number=128,
          dpsr_replay_beta_iters=None,
          dpsr_replay_beta0=0.4,
          dpsr_replay_eps=1e-6,
          dpsr_state_recycle_max_priority_set=True,
          state_recycle_freq=500,
          param_noise=False,
          callback=None,
          load_path=None,
          atari_env=True,
          **network_kwargs):
    """Train a deepq model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    network: string or a function
        neural network to use as a q function approximator. If string, has to be one of the names of registered models in baselines.common.models
        (mlp, cnn, conv_only). If a function, should take an observation tensor and return a latent variable tensor, which
        will be mapped to the Q function heads (see build_q_func in baselines.deepq.models for details on that)
    seed: int or None
        prng seed. The runs with the same seed "should" give the same results. If None, no seeding is used.
    lr: float
        learning rate for adam optimizer
    total_timesteps: int
        number of env steps to optimizer for
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
    batch_size: int
        size of a batch sampled from replay buffer for training
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    checkpoint_path: str
        the saving path of the checkpoint files
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    prioritized_replay: True
        if True prioritized replay buffer will be used.
    prioritized_replay_alpha: float
        alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        number of iterations over which beta will be annealed from initial value
        to 1.0. If set to None equals to total_timesteps.
    prioritized_replay_eps: float
        epsilon to add to the TD errors when updating priorities.
    debug_flag: bool
        if True DEBUG mode will be switched on
    dpsr_replay: bool
        if True DPSR replay buffer will be used
    dpsr_replay_alpha1: float
        alpha1 parameter for DPSR replay buffer
    dpsr_replay_alpha2: float
        alpha2 parameter for DPSR replay buffer
    dpsr_replay_candidates_size: int
        candidates size parameter for DPSR replay buffer state recycle
    dpsr_common_replacement_candidates_number: int
        candidates size parameter for DPSR replay buffer common replacement
    dpsr_replay_beta_iters: int
        number of iterations over which beta will be annealed from initial value
        to 1.0. If set to None equals to total_timesteps.
    dpsr_replay_beta0: float
        initial value of beta for prioritized replay buffer
    dpsr_replay_eps: float
        epsilon to add to the TD errors when updating priorities.
    dpsr_state_recycle_max_priority_set: bool
        if True priority will be set as MAX when doing state recycling
    state_recycle_freq: int
        do state recycling every 'state_recycle_freq' steps
    param_noise: bool
        whether or not to use parameter space noise (https://arxiv.org/abs/1706.01905)
    callback: (locals, globals) -> None
        function called at every steps with state of the algorithm.
        If callback returns true training stops.
    load_path: str
        path to load the model from. (default: None)
    atari_env: bool
        if True the env is an atari env
    **network_kwargs
        additional keyword arguments to pass to the network builder.

    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """
    # Create all the functions necessary to train the model

    sess = get_session()
    set_global_seeds(seed)

    q_func = build_q_func(network, **network_kwargs)

    # capture the shape outside the closure so that the env object is not serialized
    # by cloudpickle when serializing make_obs_ph

    observation_space = env.observation_space

    def make_obs_ph(name):
        return ObservationInput(observation_space, name=name)

    act, train, update_target, debug = deepq.build_train(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10,
        param_noise=param_noise
    )

    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'num_actions': env.action_space.n,
    }

    act = ActWrapper(act, act_params)

    # Create the replay buffer
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = total_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    elif dpsr_replay:
        replay_buffer = DoublePrioritizedStateRecycledReplayBuffer(buffer_size,
                                                                   alpha1=dpsr_replay_alpha1,
                                                                   alpha2=dpsr_replay_alpha2,
                                                                   candidates_size=dpsr_replay_candidates_size,
                                                                   # Not Used: env_id=env.env.spec.id
                                                                   env_id=None)
        if dpsr_replay_beta_iters is None:
            dpsr_replay_beta_iters = total_timesteps
        beta_schedule = LinearSchedule(dpsr_replay_beta_iters,
                                       initial_p=dpsr_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    # Initialize the parameters and copy them to the target network.
    U.initialize()
    update_target()

    episode_rewards = [0.0]
    saved_mean_reward = None
    obs = env.reset()
    reset = True

    with tempfile.TemporaryDirectory() as td:
        td = checkpoint_path or td

        model_file = os.path.join(td, "model")
        model_saved = False

        if tf.train.latest_checkpoint(td) is not None:
            load_variables(model_file)
            logger.log('Loaded model from {}'.format(model_file))
            model_saved = True
        elif load_path is not None:
            load_variables(load_path)
            logger.log('Loaded model from {}'.format(load_path))

        for t in range(total_timesteps):
            if callback is not None:
                if callback(locals(), globals()):
                    break
            # Take action and update exploration to the newest value
            kwargs = {}
            if not param_noise:
                update_eps = exploration.value(t)
                update_param_noise_threshold = 0.
            else:
                update_eps = 0.
                # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                # for detailed explanation.
                update_param_noise_threshold = -np.log(
                    1. - exploration.value(t) + exploration.value(t) / float(env.action_space.n))
                kwargs['reset'] = reset
                kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                kwargs['update_param_noise_scale'] = True
            action = act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
            env_action = action
            reset = False
            env_clone_state = None
            if dpsr_replay:
                env_clone_state = env.clone_state() if atari_env \
                    else copy.deepcopy(env.envs[0].env)
            new_obs, rew, done, _ = env.step(env_action)
            # Store transition in the replay buffer.
            if dpsr_replay:
                if replay_buffer.not_full():
                    replay_buffer.add(obs, action, rew, new_obs, float(done), env_clone_state, t)
                elif state_recycle_freq and t % state_recycle_freq == 0:
                    current_env_copy = None
                    if not atari_env:
                        current_env_copy = copy.deepcopy(env.envs[0].env)
                    candidates_idxes, candidates = replay_buffer.replacement_candidates()
                    candidates_recycled = []
                    for candidate in candidates:
                        cand_obs, cand_old_act, *_, cand_state, cand_t = candidate
                        if atari_env:
                            new_env = copy.deepcopy(env)
                            new_env.reset()
                            new_env.restore_state(cand_state)
                        else:
                            env.envs[0].env = cand_state
                        new_action_cand = act(np.array(cand_obs)[None], update_eps=0.0, **kwargs)[0]
                        # make sure that a new experience is made
                        if new_action_cand != cand_old_act:
                            new_action = new_action_cand
                        else:
                            while True:
                                new_action_cand = env.action_space.sample()
                                if new_action_cand != cand_old_act:
                                    new_action = new_action_cand
                                    break
                        if atari_env:
                            new_new_obs, new_rew, new_done, _ = new_env.step(new_action)
                        else:
                            new_new_obs, new_rew, new_done, _ = env.step(new_action)
                        new_data = (cand_obs, new_action, new_rew, new_new_obs, new_done, cand_state, t)
                        candidates_recycled.append(new_data)
                    # get the new TDEs after recycling
                    cand_obses = np.array([data[0] for data in candidates_recycled])
                    cand_acts = np.array([data[1] for data in candidates_recycled])
                    cand_rews = np.array([data[2] for data in candidates_recycled])
                    cand_new_obses = np.array([data[3] for data in candidates_recycled])
                    cand_dones = np.array([data[4] for data in candidates_recycled])
                    cand_weights = np.zeros_like(cand_rews)
                    cand_td_errors = train(cand_obses, cand_acts, cand_rews, cand_new_obses, cand_dones, cand_weights)
                    new_cand_priorities = np.abs(cand_td_errors) + dpsr_replay_eps
                    replay_buffer.update_priorities(candidates_idxes, new_cand_priorities)
                    replay_buffer.state_recycle(candidates_idxes, candidates_recycled, cand_td_errors,
                                                dpsr_state_recycle_max_priority_set)
                    replay_buffer.add(obs, action, rew, new_obs, float(done), env_clone_state, t)
                    if not atari_env:
                        env.envs[0].env = current_env_copy
                else:
                    # common_replacement_candidates_number = 128
                    candidates_idxes, candidates = replay_buffer.replacement_candidates(
                        dpsr_common_replacement_candidates_number)
                    cand_timestamps = [candidate[-1] for candidate in candidates]
                    replace_idx = candidates_idxes[np.argmin(cand_timestamps)]
                    replay_buffer.add(obs, action, rew, new_obs, float(done), env_clone_state, t, replace_idx)
            else:
                replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = env.reset()
                episode_rewards.append(0.0)
                reset = True

            if t > learning_starts and t % train_freq == 0:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if prioritized_replay:
                    experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                elif dpsr_replay:
                    experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                    (obses_t, actions, rewards, obses_tp1, dones, env_states, timestamps,
                     weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                    weights, batch_idxes = np.ones_like(rewards), None
                td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)
                if prioritized_replay:
                    new_priorities = np.abs(td_errors) + prioritized_replay_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)
                elif dpsr_replay:
                    new_priorities = np.abs(td_errors) + dpsr_replay_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)

            if t > learning_starts and t % target_network_update_freq == 0:
                # Update target network periodically.
                update_target()

            mean_100ep_reward = np.round(np.mean(episode_rewards[-101:-1]), 1)
            mean_10ep_reward = np.round(np.mean(episode_rewards[-11:-1]), 1)
            mean_5ep_reward = np.round(np.mean(episode_rewards[-6:-1]), 1)
            last_1ep_reward = np.round(np.mean(episode_rewards[-2:-1]), 1)
            num_episodes = len(episode_rewards)
            if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                logger.record_tabular("mean 10 episode reward", mean_10ep_reward)
                logger.record_tabular("mean 5 episode reward", mean_5ep_reward)
                logger.record_tabular("last episode reward", last_1ep_reward)
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()

            if (checkpoint_freq is not None and t > learning_starts and
                    num_episodes > 100 and t % checkpoint_freq == 0):
                if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                    if print_freq is not None:
                        logger.log("Saving model due to mean reward increase: {} -> {}".format(
                            saved_mean_reward, mean_100ep_reward))
                    save_variables(model_file)
                    model_saved = True
                    saved_mean_reward = mean_100ep_reward

        if model_saved:
            if print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
            load_variables(model_file)

    return act
