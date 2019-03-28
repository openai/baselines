import collections
import functools
import itertools
import multiprocessing
import pytest
import gym
import numpy as np

from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv

N_ENVS = 3
VEC_ENV_CLASSES = [DummyVecEnv, SubprocVecEnv]


class CustomGymEnv(gym.Env):
    def __init__(self, space):
        """
        Custom gym environment for testing purposes
        """
        self.action_space = space
        self.observation_space = space
        self.current_step = 0
        self.ep_length = 4

    def reset(self):
        self.current_step = 0
        self._choose_next_state()
        return self.state

    def step(self, action):
        reward = 1
        self._choose_next_state()
        self.current_step += 1
        done = self.current_step >= self.ep_length
        return self.state, reward, done, {}

    def _choose_next_state(self):
        self.state = self.observation_space.sample()

    def render(self, mode='human'):
        pass

    @staticmethod
    def custom_method(dim_0=1, dim_1=1):
        """
        Dummy method to test call to custom method
        from VecEnv

        :param dim_0: (int)
        :param dim_1: (int)
        :return: (np.ndarray)
        """
        return np.ones((dim_0, dim_1))


@pytest.mark.parametrize('vec_env_class', VEC_ENV_CLASSES)
def test_vecenv_custom_calls(vec_env_class):
    """Test access to methods/attributes of vectorized environments"""
    def make_env():
        return CustomGymEnv(gym.spaces.Discrete(2))
    vec_env = vec_env_class([make_env for _ in range(N_ENVS)])
    env_method_results = vec_env.env_method('custom_method', 1, dim_1=2)
    setattr_results = []
    # Set current_step to an arbitrary value
    for env_idx in range(N_ENVS):
        setattr_results.append(vec_env.set_attr('current_step', env_idx, indices=env_idx))
    # Retrieve the value for each environment
    getattr_results = vec_env.get_attr('current_step')

    assert len(env_method_results) == N_ENVS
    assert len(setattr_results) == N_ENVS
    assert len(getattr_results) == N_ENVS

    for env_idx in range(N_ENVS):
        assert (env_method_results[env_idx] == np.ones((1, 2))).all()
        assert setattr_results[env_idx][0] is None
        assert getattr_results[env_idx] == env_idx

    # Test to change value for all the environments
    setattr_result = vec_env.set_attr('current_step', 42, indices=None)
    getattr_result = vec_env.get_attr('current_step')
    assert setattr_result == [None for _ in range(N_ENVS)]
    assert getattr_result == [42 for _ in range(N_ENVS)]

    # Additional tests for setattr that does not affect all the environments
    vec_env.reset()
    setattr_result = vec_env.set_attr('current_step', 12, indices=[0, 1])
    getattr_result = vec_env.get_attr('current_step')
    assert setattr_result == [None for _ in range(2)]
    assert getattr_result == [12 for _ in range(2)] + [0 for _ in range(N_ENVS - 2)]

    vec_env.reset()
    # Change value only for first and last environment
    setattr_result = vec_env.set_attr('current_step', 12, indices=[0, -1])
    getattr_result = vec_env.get_attr('current_step')
    assert setattr_result == [None for _ in range(2)]
    assert getattr_result == [12] + [0 for _ in range(N_ENVS - 2)] + [12]

    vec_env.close()


SPACES = collections.OrderedDict([
    ('discrete', gym.spaces.Discrete(2)),
    ('multidiscrete', gym.spaces.MultiDiscrete([2, 3])),
    ('multibinary', gym.spaces.MultiBinary(3)),
    ('continuous', gym.spaces.Box(low=np.zeros(2), high=np.ones(2))),
])

def check_vecenv_spaces(vec_env_class, space, obs_assert):
    """Helper method to check observation spaces in vectorized environments."""
    def make_env():
        return CustomGymEnv(space)

    vec_env = vec_env_class([make_env for _ in range(N_ENVS)])
    obs = vec_env.reset()
    obs_assert(obs)

    dones = [False] * N_ENVS
    while not any(dones):
        actions = [vec_env.action_space.sample() for _ in range(N_ENVS)]
        obs, _rews, dones, _infos = vec_env.step(actions)
        obs_assert(obs)
    vec_env.close()

    vec_env.close()


def check_vecenv_obs(obs, space):
    """Helper method to check observations from multiple environments each belong to
       the appropriate observation space."""
    assert obs.shape[0] == N_ENVS
    for value in obs:
        assert space.contains(value)


@pytest.mark.parametrize('vec_env_class,space', itertools.product(VEC_ENV_CLASSES, SPACES.values()))
def test_vecenv_single_space(vec_env_class, space):
    def obs_assert(obs):
        return check_vecenv_obs(obs, space)

    check_vecenv_spaces(vec_env_class, space, obs_assert)


class _UnorderedDictSpace(gym.spaces.Dict):
    """Like DictSpace, but returns an unordered dict when sampling."""
    def sample(self):
        return dict(super().sample())


@pytest.mark.parametrize('vec_env_class', VEC_ENV_CLASSES)
def test_vecenv_dict_spaces(vec_env_class):
    """Test dictionary observation spaces with vectorized environments."""
    space = gym.spaces.Dict(SPACES)

    def obs_assert(obs):
        assert isinstance(obs, collections.OrderedDict)
        assert obs.keys() == space.spaces.keys()
        for key, values in obs.items():
            check_vecenv_obs(values, space.spaces[key])

    check_vecenv_spaces(vec_env_class, space, obs_assert)

    unordered_space = _UnorderedDictSpace(SPACES)
    # Check that vec_env_class can accept unordered dict observations (and convert to OrderedDict)
    check_vecenv_spaces(vec_env_class, unordered_space, obs_assert)


@pytest.mark.parametrize('vec_env_class', VEC_ENV_CLASSES)
def test_vecenv_tuple_spaces(vec_env_class):
    """Test tuple observation spaces with vectorized environments."""
    space = gym.spaces.Tuple(tuple(SPACES.values()))

    def obs_assert(obs):
        assert isinstance(obs, tuple)
        assert len(obs) == len(space.spaces)
        for values, inner_space in zip(obs, space.spaces):
            check_vecenv_obs(values, inner_space)

    return check_vecenv_spaces(vec_env_class, space, obs_assert)


def test_subproc_start_method():
    start_methods = [None] + multiprocessing.get_all_start_methods()
    space = gym.spaces.Discrete(2)

    def obs_assert(obs):
        return check_vecenv_obs(obs, space)

    for start_method in start_methods:
        vec_env_class = functools.partial(SubprocVecEnv, start_method=start_method)
        check_vecenv_spaces(vec_env_class, space, obs_assert)

    with pytest.raises(ValueError, match="cannot find context for 'illegal_method'"):
        vec_env_class = functools.partial(SubprocVecEnv, start_method='illegal_method')
        check_vecenv_spaces(vec_env_class, space, obs_assert)
