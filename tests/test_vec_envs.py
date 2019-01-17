import pytest
import gym
import numpy as np

from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv

N_ENVS = 3


class CustomGymEnv(gym.Env):
    def __init__(self):
        """
        Custom gym environment for testing purposes
        """
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = self.action_space
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
        self.state = self.action_space.sample()

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


@pytest.mark.parametrize("vec_env_class", [DummyVecEnv, SubprocVecEnv])
def test_vecenv_custom_calls(vec_env_class):
    """Test access to methods/attributes of vectorized environments"""
    vec_env = vec_env_class([CustomGymEnv for _ in range(N_ENVS)])
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
