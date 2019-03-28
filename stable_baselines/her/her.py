import tensorflow as tf
import numpy as np
import gym

from stable_baselines.common import BaseRLModel, SetVerbosity
from stable_baselines.common.policies import LstmPolicy, ActorCriticPolicy


def make_sample_her_transitions(replay_strategy, replay_k, reward_fun):
    """
    Creates a sample function that can be used for HER experience replay.

    :param replay_strategy: (str) the HER replay strategy; if set to 'none', regular DDPG experience replay is used
        (can be 'future' or 'none').
    :param replay_k: (int) the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
    :param reward_fun: (function (dict, dict): float) function to re-compute the reward with substituted goals
    """
    if replay_strategy == 'future':
        future_p = 1 - (1. / (1 + replay_k))
    else:  # 'replay_strategy' == 'none'
        future_p = 0

    def _sample_her_transitions(episode_batch, batch_size_in_transitions):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        time_horizon = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions

        # Select which episodes and time steps to use.
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(time_horizon, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        future_offset = np.random.uniform(size=batch_size) * (time_horizon - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        # Replace goal with achieved goal but only for the previously-selected
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag

        # Reconstruct info dictionary for reward  computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info
        transitions['r'] = reward_fun(**reward_params)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert transitions['u'].shape[0] == batch_size_in_transitions

        return transitions

    return _sample_her_transitions


class HER(BaseRLModel):
    def __init__(self, policy, env, verbose=0, _init_setup_model=True):
        super().__init__(policy=policy, env=env, verbose=verbose, policy_base=ActorCriticPolicy, requires_vec_env=False)

        self.policy = policy

        self.sess = None
        self.graph = None

        if _init_setup_model:
            self.setup_model()

    def _get_pretrain_placeholders(self):
        raise NotImplementedError()

    def setup_model(self):
        with SetVerbosity(self.verbose):

            assert isinstance(self.action_space, gym.spaces.Box), \
                "Error: HER cannot output a {} action space, only spaces.Box is supported.".format(self.action_space)
            assert not issubclass(self.policy, LstmPolicy), "Error: cannot use a recurrent policy for the HER model."
            assert issubclass(self.policy, ActorCriticPolicy), "Error: the input policy for the HER model must be an " \
                                                               "instance of common.policies.ActorCriticPolicy."

            self.graph = tf.Graph()
            with self.graph.as_default():
                pass

    def learn(self, total_timesteps, callback=None, seed=None, log_interval=100, tb_log_name="HER",
              reset_num_timesteps=True):
        with SetVerbosity(self.verbose):
            self._setup_learn(seed)

        return self

    def predict(self, observation, state=None, mask=None, deterministic=False):
        pass

    def action_probability(self, observation, state=None, mask=None, actions=None):
        pass

    def save(self, save_path):
        pass

    @classmethod
    def load(cls, load_path, env=None, **kwargs):
        pass
