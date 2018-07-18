from abc import ABC, abstractmethod


class BaseRLModel(ABC):
    def __init__(self):
        super(BaseRLModel, self).__init__()

    @abstractmethod
    def learn(self, callback=None, seed=None):
        """
        Return a trained model.

        :param seed: (int) The initial seed for training, if None: keep current seed
        :param callback: (function (dict, dict)) function called at every steps with state of the algorithm.
            It takes the local and global variables.
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

    @abstractmethod
    def load(self, load_path):
        """
        Load the parameters from file

        :param load_path: (str) the saved parameter location
        """
        pass
