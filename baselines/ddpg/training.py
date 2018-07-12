import os
import time
from collections import deque
import pickle

import numpy as np
import tensorflow as tf
from mpi4py import MPI

from baselines.ddpg.ddpg import DDPG
import baselines.common.tf_util as tf_util
from baselines import logger


def train(env, nb_epochs, nb_epoch_cycles, render_eval, reward_scale, render, param_noise, actor, critic,
          normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr, action_noise,
          popart, gamma, clip_norm, nb_train_steps, nb_rollout_steps, nb_eval_steps, batch_size, memory,
          tau=0.01, eval_env=None, param_noise_adaption_interval=50):
    """
    Runs the training of the Deep Deterministic Policy Gradien (DDPG) model

    DDPG: https://arxiv.org/pdf/1509.02971.pdf

    :param env: (Gym Environment) the environment
    :param nb_epochs: (int) the number of training epochs
    :param nb_epoch_cycles: (int) the number cycles within each epoch
    :param render_eval: (bool) enable rendering of the evalution environment
    :param reward_scale: (float) the value the reward should be scaled by
    :param render: (bool) enable rendering of the environment
    :param param_noise: (AdaptiveParamNoiseSpec) the parameter noise type (can be None)
    :param actor: (TensorFlow Tensor) the actor model
    :param critic: (TensorFlow Tensor) the critic model
    :param normalize_returns: (bool) should the critic output be normalized
    :param normalize_observations: (bool) should the observation be normalized
    :param critic_l2_reg: (float) l2 regularizer coefficient
    :param actor_lr: (float) the actor learning rate
    :param critic_lr: (float) the critic learning rate
    :param action_noise: (ActionNoise) the action noise type (can be None)
    :param popart: (bool) enable pop-art normalization of the critic output
        (https://arxiv.org/pdf/1602.07714.pdf)
    :param gamma: (float) the discount rate
    :param clip_norm: (float) clip the gradiants (disabled if None)
    :param nb_train_steps: (int) the number of training steps
    :param nb_rollout_steps: (int) the number of rollout steps
    :param nb_eval_steps: (int) the number of evalutation steps
    :param batch_size: (int) the size of the batch for learning the policy
    :param memory: (Memory) the replay buffer
    :param tau: (float) the soft update coefficient (keep old values, between 0 and 1)
    :param eval_env: (Gym Environment) the evaluation environment (can be None)
    :param param_noise_adaption_interval: (int) apply param noise every N steps
    """
    rank = MPI.COMM_WORLD.Get_rank()

    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    max_action = env.action_space.high
    logger.info('scaling actions by {} before executing in env'.format(max_action))
    agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape, gamma=gamma, tau=tau,
                 normalize_returns=normalize_returns, normalize_observations=normalize_observations,
                 batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
                 actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
                 reward_scale=reward_scale)
    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))

    # Set up logging stuff only for a single worker.
    if rank == 0:
        tf.train.Saver()

    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)
    with tf_util.single_threaded_session() as sess:
        # Prepare everything.
        agent.initialize(sess)
        sess.graph.finalize()

        agent.reset()
        obs = env.reset()
        if eval_env is not None:
            eval_obs = eval_env.reset()
        episode_reward = 0.
        episode_step = 0
        episodes = 0
        t = 0

        start_time = time.time()

        epoch_episode_rewards = []
        epoch_episode_steps = []
        epoch_actions = []
        epoch_qs = []
        epoch_episodes = 0
        for epoch in range(nb_epochs):
            for cycle in range(nb_epoch_cycles):
                # Perform rollouts.
                for t_rollout in range(nb_rollout_steps):
                    # Predict next action.
                    action, q = agent.pi(obs, apply_noise=True, compute_Q=True)
                    assert action.shape == env.action_space.shape

                    # Execute next action.
                    if rank == 0 and render:
                        env.render()
                    assert max_action.shape == action.shape
                    # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                    new_obs, r, done, info = env.step(max_action * action)
                    t += 1
                    if rank == 0 and render:
                        env.render()
                    episode_reward += r
                    episode_step += 1

                    # Book-keeping.
                    epoch_actions.append(action)
                    epoch_qs.append(q)
                    agent.store_transition(obs, action, r, new_obs, done)
                    obs = new_obs

                    if done:
                        # Episode done.
                        epoch_episode_rewards.append(episode_reward)
                        episode_rewards_history.append(episode_reward)
                        epoch_episode_steps.append(episode_step)
                        episode_reward = 0.
                        episode_step = 0
                        epoch_episodes += 1
                        episodes += 1

                        agent.reset()
                        obs = env.reset()

                # Train.
                epoch_actor_losses = []
                epoch_critic_losses = []
                epoch_adaptive_distances = []
                for t_train in range(nb_train_steps):
                    # Adapt param noise, if necessary.
                    if memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
                        distance = agent.adapt_param_noise()
                        epoch_adaptive_distances.append(distance)

                    cl, al = agent.train()
                    epoch_critic_losses.append(cl)
                    epoch_actor_losses.append(al)
                    agent.update_target_net()

                # Evaluate.
                eval_episode_rewards = []
                eval_qs = []
                if eval_env is not None:
                    eval_episode_reward = 0.
                    for t_rollout in range(nb_eval_steps):
                        eval_action, eval_q = agent.pi(eval_obs, apply_noise=False, compute_Q=True)
                        # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                        eval_obs, eval_r, eval_done, eval_info = eval_env.step(max_action * eval_action)
                        if render_eval:
                            eval_env.render()
                        eval_episode_reward += eval_r

                        eval_qs.append(eval_q)
                        if eval_done:
                            eval_obs = eval_env.reset()
                            eval_episode_rewards.append(eval_episode_reward)
                            eval_episode_rewards_history.append(eval_episode_reward)
                            eval_episode_reward = 0.

            mpi_size = MPI.COMM_WORLD.Get_size()
            # Log stats.
            # XXX shouldn't call np.mean on variable length lists
            duration = time.time() - start_time
            stats = agent.get_stats()
            combined_stats = stats.copy()
            combined_stats['rollout/return'] = np.mean(epoch_episode_rewards)
            combined_stats['rollout/return_history'] = np.mean(episode_rewards_history)
            combined_stats['rollout/episode_steps'] = np.mean(epoch_episode_steps)
            combined_stats['rollout/actions_mean'] = np.mean(epoch_actions)
            combined_stats['rollout/Q_mean'] = np.mean(epoch_qs)
            combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
            combined_stats['train/loss_critic'] = np.mean(epoch_critic_losses)
            combined_stats['train/param_noise_distance'] = np.mean(epoch_adaptive_distances)
            combined_stats['total/duration'] = duration
            combined_stats['total/steps_per_second'] = float(t) / float(duration)
            combined_stats['total/episodes'] = episodes
            combined_stats['rollout/episodes'] = epoch_episodes
            combined_stats['rollout/actions_std'] = np.std(epoch_actions)
            # Evaluation statistics.
            if eval_env is not None:
                combined_stats['eval/return'] = eval_episode_rewards
                combined_stats['eval/return_history'] = np.mean(eval_episode_rewards_history)
                combined_stats['eval/Q'] = eval_qs
                combined_stats['eval/episodes'] = len(eval_episode_rewards)

            def as_scalar(x):
                """
                check and return the input if it is a scalar, otherwise raise ValueError

                :param x: (Any) the object to check
                :return: (Number) the scalar if x is a scalar
                """
                if isinstance(x, np.ndarray):
                    assert x.size == 1
                    return x[0]
                elif np.isscalar(x):
                    return x
                else:
                    raise ValueError('expected scalar, got %s' % x)
            combined_stats_sums = MPI.COMM_WORLD.allreduce(np.array([as_scalar(x) for x in combined_stats.values()]))
            combined_stats = {k: v / mpi_size for (k, v) in zip(combined_stats.keys(), combined_stats_sums)}

            # Total statistics.
            combined_stats['total/epochs'] = epoch + 1
            combined_stats['total/steps'] = t

            for key in sorted(combined_stats.keys()):
                logger.record_tabular(key, combined_stats[key])
            logger.dump_tabular()
            logger.info('')
            logdir = logger.get_dir()
            if rank == 0 and logdir:
                if hasattr(env, 'get_state'):
                    with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
                        pickle.dump(env.get_state(), f)
                if eval_env and hasattr(eval_env, 'get_state'):
                    with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as f:
                        pickle.dump(eval_env.get_state(), f)
