import numpy as np
from abc import ABC, abstractmethod


class AbstractEnvRunner(ABC):
    def __init__(self, *, env, model, n_steps):
        """
        A runner to learn the policy of an environment for a model

        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        """
        self.env = env
        self.model = model
        n_env = env.num_envs
        self.batch_ob_shape = (n_env*n_steps,) + env.observation_space.shape
        self.obs = np.zeros((n_env,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
        self.obs[:] = env.reset()
        self.n_steps = n_steps
        self.states = model.initial_state
        self.dones = [False for _ in range(n_env)]

    @abstractmethod
    def run(self):
        """
        Run a learning step of the model
        """
        raise NotImplementedError
