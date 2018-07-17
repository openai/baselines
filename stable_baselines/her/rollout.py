from collections import deque
import pickle

import numpy as np
from mujoco_py import MujocoException

from stable_baselines.her.util import convert_episode_to_batch_major


class RolloutWorker:
    def __init__(self, make_env, policy, dims, logger, time_horizon, rollout_batch_size=1,
                 exploit=False, use_target_net=False, compute_q=False, noise_eps=0,
                 random_eps=0, history_len=100, render=False):
        """
        Rollout worker generates experience by interacting with one or many environments.

        :param make_env: (function (): Gym Environment) a factory function that creates a new instance of the
            environment when called
        :param policy: (Object) the policy that is used to act
        :param dims: ({str: int}) the dimensions for observations (o), goals (g), and actions (u)
        :param logger: (Object) the logger that is used by the rollout worker
        :param rollout_batch_size: (int) the number of parallel rollouts that should be used
        :param exploit: (bool) whether or not to exploit, i.e. to act optimally according to the current policy without
            any exploration
        :param use_target_net: (bool) whether or not to use the target net for rollouts
        :param compute_q: (bool) whether or not to compute the Q values alongside the actions
        :param noise_eps: (float) scale of the additive Gaussian noise
        :param random_eps: (float) probability of selecting a completely random action
        :param history_len: (int) length of history for statistics smoothing
        :param render: (boolean) whether or not to render the rollouts
        """
        self.make_env = make_env
        self.policy = policy
        self.dims = dims
        self.logger = logger
        self.time_horizon = time_horizon
        self.rollout_batch_size = rollout_batch_size
        self.exploit = exploit
        self.use_target_net = use_target_net
        self.compute_q = compute_q
        self.noise_eps = noise_eps
        self.random_eps = random_eps
        self.history_len = history_len
        self.render = render

        self.envs = [make_env() for _ in range(rollout_batch_size)]
        assert self.time_horizon > 0

        self.info_keys = [key.replace('info_', '') for key in dims.keys() if key.startswith('info_')]

        self.success_history = deque(maxlen=history_len)
        self.q_history = deque(maxlen=history_len)

        self.n_episodes = 0
        self.goals = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # goals
        self.initial_obs = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        self.initial_ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        self.reset_all_rollouts()
        self.clear_history()

    def reset_rollout(self, index):
        """
        Resets the `i`-th rollout environment, re-samples a new goal, and updates the `initial_o` and `g` arrays
        accordingly.

        :param index: (int) the index to reset
        """
        obs = self.envs[index].reset()
        self.initial_obs[index] = obs['observation']
        self.initial_ag[index] = obs['achieved_goal']
        self.goals[index] = obs['desired_goal']

    def reset_all_rollouts(self):
        """
        Resets all `rollout_batch_size` rollout workers.
        """
        for step in range(self.rollout_batch_size):
            self.reset_rollout(step)

    def generate_rollouts(self):
        """
        Performs `rollout_batch_size` rollouts in parallel for time horizon with the current
        policy acting on it accordingly.

        :return: (dict) batch
        """
        self.reset_all_rollouts()

        # compute observations
        observations = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        achieved_goals = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        observations[:] = self.initial_obs
        achieved_goals[:] = self.initial_ag

        # generate episodes
        obs, achieved_goals, acts, goals, successes = [], [], [], [], []
        info_values = [np.empty((self.time_horizon, self.rollout_batch_size, self.dims['info_' + key]), np.float32)
                       for key in self.info_keys]
        q_values = []
        for step in range(self.time_horizon):
            policy_output = self.policy.get_actions(
                observations, achieved_goals, self.goals,
                compute_q=self.compute_q,
                noise_eps=self.noise_eps if not self.exploit else 0.,
                random_eps=self.random_eps if not self.exploit else 0.,
                use_target_net=self.use_target_net)

            if self.compute_q:
                action, q_value = policy_output
                q_values.append(q_value)
            else:
                action = policy_output

            if action.ndim == 1:
                # The non-batched case should still have a reasonable shape.
                action = action.reshape(1, -1)

            o_new = np.empty((self.rollout_batch_size, self.dims['o']))
            ag_new = np.empty((self.rollout_batch_size, self.dims['g']))
            success = np.zeros(self.rollout_batch_size)
            # compute new states and observations
            for batch_idx in range(self.rollout_batch_size):
                try:
                    # We fully ignore the reward here because it will have to be re-computed
                    # for HER.
                    curr_o_new, _, _, info = self.envs[batch_idx].step(action[batch_idx])
                    if 'is_success' in info:
                        success[batch_idx] = info['is_success']
                    o_new[batch_idx] = curr_o_new['observation']
                    ag_new[batch_idx] = curr_o_new['achieved_goal']
                    for idx, key in enumerate(self.info_keys):
                        info_values[idx][step, batch_idx] = info[key]
                    if self.render:
                        self.envs[batch_idx].render()
                except MujocoException:
                    return self.generate_rollouts()

            if np.isnan(o_new).any():
                self.logger.warning('NaN caught during rollout generation. Trying again...')
                self.reset_all_rollouts()
                return self.generate_rollouts()

            obs.append(observations.copy())
            achieved_goals.append(achieved_goals.copy())
            successes.append(success.copy())
            acts.append(action.copy())
            goals.append(self.goals.copy())
            observations[...] = o_new
            achieved_goals[...] = ag_new
        obs.append(observations.copy())
        achieved_goals.append(achieved_goals.copy())
        self.initial_obs[:] = observations

        episode = dict(o=obs,
                       u=acts,
                       g=goals,
                       ag=achieved_goals)
        for key, value in zip(self.info_keys, info_values):
            episode['info_{}'.format(key)] = value

        # stats
        successful = np.array(successes)[-1, :]
        assert successful.shape == (self.rollout_batch_size,)
        success_rate = np.mean(successful)
        self.success_history.append(success_rate)

        if self.compute_q:
            self.q_history.append(np.mean(q_values))
        self.n_episodes += self.rollout_batch_size

        return convert_episode_to_batch_major(episode)

    def clear_history(self):
        """
        Clears all histories that are used for statistics
        """
        self.success_history.clear()
        self.q_history.clear()

    def current_success_rate(self):
        """
        returns the current success rate
        :return: (float) the success rate
        """
        return np.mean(self.success_history)

    def current_mean_q(self):
        """
        returns the current mean Q value
        :return: (float) the mean Q value
        """
        return np.mean(self.q_history)

    def save_policy(self, path):
        """
        Pickles the current policy for later inspection.

        :param path: (str) the save location
        """
        with open(path, 'wb') as file_handler:
            pickle.dump(self.policy, file_handler)

    def logs(self, prefix='worker'):
        """
        Generates a dictionary that contains all collected statistics.

        :param prefix: (str) the prefix for the name in logging
        :return: ([(str, float)]) the logging information
        """
        logs = []
        logs += [('success_rate', np.mean(self.success_history))]
        if self.compute_q:
            logs += [('mean_q', np.mean(self.q_history))]
        logs += [('episode', self.n_episodes)]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def seed(self, seed):
        """
        Seeds each environment with a distinct seed derived from the passed in global seed.

        :param seed: (int) the random seed
        """
        for idx, env in enumerate(self.envs):
            env.seed(seed + 1000 * idx)
