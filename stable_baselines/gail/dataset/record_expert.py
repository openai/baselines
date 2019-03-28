import os
import warnings

import cv2
import numpy as np
from gym import spaces

from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.vec_env import VecEnv, VecFrameStack


def generate_expert_traj(model, save_path, env=None, n_timesteps=0,
                         n_episodes=100, image_folder='recorded_images'):
    """
    Train expert controller (if needed) and record expert trajectories.

    .. note::

        only Box and Discrete spaces are supported for now.

    :param model: (RL model or callable) The expert model, if it needs to be trained,
        then you need to pass ``n_timesteps > 0``.
    :param save_path: (str) Path without the extension where the
        expert dataset will be saved (ex: 'expert_cartpole' -> creates 'expert_cartpole.npz')
    :param env: (gym.Env) The environment, if not defined then it tries to use the model
        environment.
    :param n_timesteps: (int) Number of training timesteps
    :param n_episodes: (int) Number of trajectories (episodes) to record
    :param image_folder: (str) When using images, folder that will be used to record images.
    """

    # Retrieve the environment using the RL model
    if env is None and isinstance(model, BaseRLModel):
        env = model.get_env()

    assert env is not None, "You must set the env in the model or pass it to the function."

    is_vec_env = False
    if isinstance(env, VecEnv):
        is_vec_env = True
        assert env.num_envs == 1, 'You must use only one env to record expert data'

    # Sanity check
    assert (isinstance(env.observation_space, spaces.Box) or
            isinstance(env.observation_space, spaces.Discrete)), "Observation space type not supported"

    assert (isinstance(env.action_space, spaces.Box) or
            isinstance(env.action_space, spaces.Discrete)), "Action space type not supported"

    # Check if we need to record images
    obs_space = env.observation_space
    record_images = len(obs_space.shape) == 3 and obs_space.shape[-1] in [1, 3, 4] \
                    and obs_space.dtype == np.uint8

    if not record_images and len(obs_space.shape) == 3 and obs_space.dtype == np.uint8:
        warnings.warn("The observations looks like images (shape = {}) "
                      "but the number of channel > 4, so it will be saved in the numpy archive "
                      "which can lead to high memory usage".format(obs_space.shape))

    image_ext = 'jpg'
    if record_images:
        # We save images as jpg or png, that have only 3/4 color channels
        if isinstance(env, VecFrameStack) and env.n_stack == 4:
            # assert env.n_stack < 5, "The current data recorder does no support"\
            #                          "VecFrameStack with n_stack > 4"
            image_ext = 'png'

        folder_path = os.path.dirname(save_path)
        image_folder = os.path.join(folder_path, image_folder)
        os.makedirs(image_folder, exist_ok=True)
        print("=" * 10)
        print("Images will be recorded to {}/".format(image_folder))
        print("Image shape: {}".format(obs_space.shape))
        print("=" * 10)

    if n_timesteps > 0 and isinstance(model, BaseRLModel):
        model.learn(n_timesteps)

    actions = []
    observations = []
    rewards = []
    episode_returns = np.zeros((n_episodes,))
    episode_starts = []

    ep_idx = 0
    obs = env.reset()
    episode_starts.append(True)
    reward_sum = 0.0
    idx = 0
    while ep_idx < n_episodes:
        if record_images:
            image_path = os.path.join(image_folder, "{}.{}".format(idx, image_ext))
            obs_ = obs[0] if is_vec_env else obs
            # Convert from RGB to BGR
            # which is the format OpenCV expect
            if obs_.shape[-1] == 3:
                obs_ = cv2.cvtColor(obs_, cv2.COLOR_RGB2BGR)
            cv2.imwrite(image_path, obs_)
            observations.append(image_path)
        else:
            observations.append(obs)

        if isinstance(model, BaseRLModel):
            action, _ = model.predict(obs)
        else:
            action = model(obs)

        obs, reward, done, _ = env.step(action)

        actions.append(action)
        rewards.append(reward)
        episode_starts.append(done)
        reward_sum += reward
        idx += 1
        if done:
            obs = env.reset()
            episode_returns[ep_idx] = reward_sum
            reward_sum = 0.0
            ep_idx += 1

    if isinstance(env.observation_space, spaces.Box) and not record_images:
        observations = np.concatenate(observations).reshape((-1,) + env.observation_space.shape)
    elif isinstance(env.observation_space, spaces.Discrete):
        observations = np.array(observations).reshape((-1, 1))
    elif record_images:
        observations = np.array(observations)

    if isinstance(env.action_space, spaces.Box):
        actions = np.concatenate(actions).reshape((-1,) + env.action_space.shape)
    elif isinstance(env.action_space, spaces.Discrete):
        actions = np.array(actions).reshape((-1, 1))

    rewards = np.array(rewards)
    episode_starts = np.array(episode_starts[:-1])

    assert len(observations) == len(actions)

    numpy_dict = {
        'actions': actions,
        'obs': observations,
        'rewards': rewards,
        'episode_returns': episode_returns,
        'episode_starts': episode_starts
    }

    for key, val in numpy_dict.items():
        print(key, val.shape)

    np.savez(save_path, **numpy_dict)
