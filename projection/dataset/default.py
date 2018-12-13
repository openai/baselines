import os
import numpy as np

from logging import getLogger, basicConfig, DEBUG
logger = getLogger(__name__)

# Chainer
import chainer

# ---------------------------------------------------------
class DefaultDataset(chainer.dataset.DatasetMixin):
    """ Default Dataset Object for Multi-Class Classification
    """

    def __init__(self, X, Y_key, labels,):
        """
        Args.
        -----
        - X      : Input np.ndarray, shape= [Sample, Timestep, Features], dtype=float32
        - Y_key  : Label [sample,]
        - labels : Dictionary of labels
        """
        assert X.dtype == np.float32, "Invaild data type for X. expected np.float32 but get {}".format(X.dtype)
        assert Y_key.dtype == np.int32,   "Invaild data type for Y. expected np.float32 but get {}".format(Y.dtype)
        self.Y_key = Y_key
        self.labels = labels

        # Convert X to channel First
        _X = X.transpose(0,2,1)
        logger.debug("_X= {} => {}".format(X.shape, _X.shape))
        sh = _X.shape
        _X = X.reshape((sh[0], sh[1], sh[2], 1)) # STC => SCHT
        self.X = _X
        logger.debug("self.X = {}".format(self.X.shape))

        # Renew Label (Raw => Sequential Number)
        self.Y = to_sequential_number_label(self.Y_key, labels)
        
        
    def __len__(self):
        return len(self.X)
    
    def get_example(self, i):
        return self.X[i], self.Y[i]
