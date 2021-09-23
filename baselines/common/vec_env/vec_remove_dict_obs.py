from .vec_env import VecEnvObservationWrapper

class VecExtractDictObs(VecEnvObservationWrapper):
    def __init__(self, venv, key):
        self.key = key
        super().__init__(venv=venv,
            observation_space=venv.observation_space.spaces[self.key])

    def process(self, obs):
        return obs[self.key]
