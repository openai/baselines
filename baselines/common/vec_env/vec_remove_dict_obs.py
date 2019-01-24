from .vec_env import VecEnvWrapper


class VecRemoveDictObs(VecEnvWrapper):
    """
    PPO2 doesn't support dictionary observations, so make the environment only expose the observation for the provided key.
    """

    def __init__(self, venv, key):
        self._key = key
        self._venv = venv
        super().__init__(venv, observation_space=venv.observation_space.spaces[self._key])

    def _remove_dict(self, obs):
        return obs[self._key]

    def reset(self):
        return self._remove_dict(self._venv.reset())

    def step_wait(self):
        obs, rews, dones, infos = self._venv.step_wait()
        return self._remove_dict(obs), rews, dones, infos