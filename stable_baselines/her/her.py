import functools

from stable_baselines.common import BaseRLModel
from stable_baselines.common import OffPolicyRLModel
from stable_baselines.common.base_class import _UnvecWrapper
from stable_baselines.common.vec_env import VecEnvWrapper
from .replay_buffer import HindsightExperienceReplayWrapper, KEY_TO_GOAL_STRATEGY
from .utils import HERGoalEnvWrapper


class HER(BaseRLModel):
    """
    Hindsight Experience Replay (HER) https://arxiv.org/abs/1707.01495

    :param policy: (BasePolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param model_class: (OffPolicyRLModel) The off policy RL model to apply Hindsight Experience Replay
        currently supported: DQN, DDPG, SAC
    :param n_sampled_goal: (int)
    :param goal_selection_strategy: (GoalSelectionStrategy or str)
    """

    def __init__(self, policy, env, model_class, n_sampled_goal=4,
                 goal_selection_strategy='future', *args, **kwargs):

        assert not isinstance(env, VecEnvWrapper), "HER does not support VecEnvWrapper"

        super().__init__(policy=policy, env=env, verbose=kwargs.get('verbose', 0),
                         policy_base=None, requires_vec_env=False)

        self.model_class = model_class
        self.replay_wrapper = None
        # Save dict observation space (used for checks at loading time)
        if env is not None:
            self.observation_space = env.observation_space
            self.action_space = env.action_space

        # Convert string to GoalSelectionStrategy object
        if isinstance(goal_selection_strategy, str):
            assert goal_selection_strategy in KEY_TO_GOAL_STRATEGY.keys(), "Unknown goal selection strategy"
            goal_selection_strategy = KEY_TO_GOAL_STRATEGY[goal_selection_strategy]

        self.n_sampled_goal = n_sampled_goal
        self.goal_selection_strategy = goal_selection_strategy

        if self.env is not None:
            self._create_replay_wrapper(self.env)

        assert issubclass(model_class, OffPolicyRLModel), \
            "Error: HER only works with Off policy model (such as DDPG, SAC, TD3 and DQN)."

        self.model = self.model_class(policy, self.env, *args, **kwargs)
        # Patch to support saving/loading
        self.model._save_to_file = self._save_to_file

    def _create_replay_wrapper(self, env):
        """
        Wrap the environment in a HERGoalEnvWrapper
        if needed and create the replay buffer wrapper.
        """
        if not isinstance(env, HERGoalEnvWrapper):
            env = HERGoalEnvWrapper(env)

        self.env = env
        # NOTE: we cannot do that check directly with VecEnv
        # maybe we can try calling `compute_reward()` ?
        # assert isinstance(self.env, gym.GoalEnv), "HER only supports gym.GoalEnv"

        self.replay_wrapper = functools.partial(HindsightExperienceReplayWrapper,
                                                n_sampled_goal=self.n_sampled_goal,
                                                goal_selection_strategy=self.goal_selection_strategy,
                                                wrapped_env=self.env)

    def set_env(self, env):
        assert not isinstance(env, VecEnvWrapper), "HER does not support VecEnvWrapper"
        super().set_env(env)
        self._create_replay_wrapper(self.env)
        self.model.set_env(self.env)

    def get_env(self):
        return self.env

    def get_parameter_list(self):
        return self.model.get_parameter_list()

    def __getattr__(self, attr):
        """
        Wrap the RL model.

        :param attr: (str)
        :return: (Any)
        """
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.model, attr)

    def __set_attr__(self, attr, value):
        if attr in self.__dict__:
            setattr(self, attr, value)
        else:
            setattr(self.model, attr, value)

    def _get_pretrain_placeholders(self):
        return self.model._get_pretrain_placeholders()

    def setup_model(self):
        pass

    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="HER",
              reset_num_timesteps=True):
        return self.model.learn(total_timesteps, callback=callback, log_interval=log_interval,
                                tb_log_name=tb_log_name, reset_num_timesteps=reset_num_timesteps,
                                replay_wrapper=self.replay_wrapper)

    def _check_obs(self, observation):
        if isinstance(observation, dict):
            if self.env is not None:
                if len(observation['observation'].shape) > 1:
                    observation = _UnvecWrapper.unvec_obs(observation)
                    return [self.env.convert_dict_to_obs(observation)]
                return self.env.convert_dict_to_obs(observation)
            else:
                raise ValueError("You must either pass an env to HER or wrap your env using HERGoalEnvWrapper")
        return observation

    def predict(self, observation, state=None, mask=None, deterministic=True):
        return self.model.predict(self._check_obs(observation), state, mask, deterministic)

    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        return self.model.action_probability(self._check_obs(observation), state, mask, actions, logp)

    def _save_to_file(self, save_path, data=None, params=None, cloudpickle=False):
        # HACK to save the replay wrapper
        # or better to save only the replay strategy and its params?
        # it will not work with VecEnv
        data['n_sampled_goal'] = self.n_sampled_goal
        data['goal_selection_strategy'] = self.goal_selection_strategy
        data['model_class'] = self.model_class
        data['her_obs_space'] = self.observation_space
        data['her_action_space'] = self.action_space
        super()._save_to_file(save_path, data, params, cloudpickle=cloudpickle)

    def save(self, save_path, cloudpickle=False):
        self.model.save(save_path, cloudpickle=cloudpickle)

    @classmethod
    def load(cls, load_path, env=None, custom_objects=None, **kwargs):
        data, _ = cls._load_from_file(load_path, custom_objects=custom_objects)

        if 'policy_kwargs' in kwargs and kwargs['policy_kwargs'] != data['policy_kwargs']:
            raise ValueError("The specified policy kwargs do not equal the stored policy kwargs. "
                             "Stored kwargs: {}, specified kwargs: {}".format(data['policy_kwargs'],
                                                                              kwargs['policy_kwargs']))

        model = cls(policy=data["policy"], env=env, model_class=data['model_class'],
                    n_sampled_goal=data['n_sampled_goal'],
                    goal_selection_strategy=data['goal_selection_strategy'],
                    _init_setup_model=False)
        model.__dict__['observation_space'] = data['her_obs_space']
        model.__dict__['action_space'] = data['her_action_space']
        model.model = data['model_class'].load(load_path, model.get_env(), **kwargs)
        model.model._save_to_file = model._save_to_file
        return model
