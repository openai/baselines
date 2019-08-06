import os
import os.path as osp

import tensorflow as tf
import numpy as np

from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines.common.vec_env.vec_env import VecEnv
from baselines.common import set_global_seeds

from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

from baselines.deepq.models import build_q_func



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
          param_noise=False,
          callback=None,
          load_path=None,
          **network_kwargs
            ):
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
        set to None to disable printing
    batch_size: int
        size of a batched sampled from replay buffer for training
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
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
    param_noise: bool
        whether or not to use parameter space noise (https://arxiv.org/abs/1706.01905)
    callback: (locals, globals) -> None
        function called at every steps with state of the algorithm.
        If callback returns true training stops.
    load_path: str
        path to load the model from. (default: None)
    **network_kwargs
        additional keyword arguments to pass to the network builder.

    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """
    # Create all the functions necessary to train the model

    set_global_seeds(seed)

    q_func = build_q_func(network, **network_kwargs)

    # capture the shape outside the closure so that the env object is not serialized
    # by cloudpickle when serializing make_obs_ph

    observation_space = env.observation_space

    model = deepq.DEEPQ(
        q_func=q_func,
        observation_shape=env.observation_space.shape,
        num_actions=env.action_space.n,
        lr=lr,
        grad_norm_clipping=10,
        gamma=gamma,
        param_noise=param_noise
    )

    if load_path is not None:
        load_path = osp.expanduser(load_path)
        ckpt = tf.train.Checkpoint(model=model)
        manager = tf.train.CheckpointManager(ckpt, load_path, max_to_keep=None)
        ckpt.restore(manager.latest_checkpoint)
        print("Restoring from {}".format(manager.latest_checkpoint))

    # Create the replay buffer
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = total_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    model.update_target()

    episode_rewards = [0.0]
    saved_mean_reward = None
    obs = env.reset()
    # always mimic the vectorized env
    if not isinstance(env, VecEnv):
        obs = np.expand_dims(np.array(obs), axis=0)
    reset = True

    for t in range(total_timesteps):
        if callback is not None:
            if callback(locals(), globals()):
                break
        kwargs = {}
        if not param_noise:
            update_eps = tf.constant(exploration.value(t))
            update_param_noise_threshold = 0.
        else:
            update_eps = tf.constant(0.)
            # Compute the threshold such that the KL divergence between perturbed and non-perturbed
            # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
            # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
            # for detailed explanation.
            update_param_noise_threshold = -np.log(1. - exploration.value(t) + exploration.value(t) / float(env.action_space.n))
            kwargs['reset'] = reset
            kwargs['update_param_noise_threshold'] = update_param_noise_threshold
            kwargs['update_param_noise_scale'] = True
        action, _, _, _ = model.step(tf.constant(obs), update_eps=update_eps, **kwargs)
        action = action[0].numpy()
        reset = False
        new_obs, rew, done, _ = env.step(action)
        # Store transition in the replay buffer.
        if not isinstance(env, VecEnv):
            new_obs = np.expand_dims(np.array(new_obs), axis=0)
            replay_buffer.add(obs[0], action, rew, new_obs[0], float(done))
        else:
            replay_buffer.add(obs[0], action, rew[0], new_obs[0], float(done[0]))
        # # Store transition in the replay buffer.
        # replay_buffer.add(obs, action, rew, new_obs, float(done))
        obs = new_obs

        episode_rewards[-1] += rew
        if done:
            obs = env.reset()
            if not isinstance(env, VecEnv):
                obs = np.expand_dims(np.array(obs), axis=0)
            episode_rewards.append(0.0)
            reset = True

        if t > learning_starts and t % train_freq == 0:
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            if prioritized_replay:
                experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
            else:
                obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                weights, batch_idxes = np.ones_like(rewards), None
            obses_t, obses_tp1 = tf.constant(obses_t), tf.constant(obses_tp1)
            actions, rewards, dones = tf.constant(actions), tf.constant(rewards), tf.constant(dones)
            weights = tf.constant(weights)
            td_errors = model.train(obses_t, actions, rewards, obses_tp1, dones, weights)
            if prioritized_replay:
                new_priorities = np.abs(td_errors) + prioritized_replay_eps
                replay_buffer.update_priorities(batch_idxes, new_priorities)

        if t > learning_starts and t % target_network_update_freq == 0:
            # Update target network periodically.
            model.update_target()

        mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
        num_episodes = len(episode_rewards)
        if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
            logger.record_tabular("steps", t)
            logger.record_tabular("episodes", num_episodes)
            logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
            logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
            logger.dump_tabular()

    return model
