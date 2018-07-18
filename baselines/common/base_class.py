from abc import ABC, abstractmethod


class BaseRLModel(ABC):
    def __init__(self):
        super(BaseRLModel, self).__init__()

    @abstractmethod
    def learn(self):
        """
        Return a trained model.

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
