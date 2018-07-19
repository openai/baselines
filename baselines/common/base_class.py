from abc import ABC, abstractmethod


class BaseRLModel(ABC):
    def __init__(self):
        super(BaseRLModel, self).__init__()

    @abstractmethod
    def learn(self, callback=None, seed=None, log_interval=100):
        """
        Return a trained model.

        :param seed: (int) The initial seed for training, if None: keep current seed
        :param callback: (function (dict, dict)) function called at every steps with state of the algorithm.
            It takes the local and global variables.
        :param log_interval: (int) The number of timesteps before logging.
        :return: (BaseRLModel) the trained model
        """
        pass

    @abstractmethod
    def save(self, save_path):
        """
        Save the current parameters to file

        :param save_path: (str) the save location
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, load_path, env):
        """
        Load the model from file

        :param load_path: (str) the saved parameter location
        :param env: (Gym Envrionment) the new environment to run the loaded model on
        """
        pass
