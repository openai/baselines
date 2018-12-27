import os
import numpy as np
import h5py


from logging import getLogger, basicConfig, DEBUG
logger = getLogger(__name__)

# Chainer
import chainer

# ---------------------------------------------------------
class DefaultDataset(chainer.dataset.DatasetMixin):
    """ Default Dataset Object for Multi-Class Classification
    """

    def __init__(self, file_list):
        """
        Args.
        -----
        - file_list :list of input files (+.h5)
        """
        X, Y = [], []
        for path in file_list:
            if not os.path.exists(path):
                logger.warning("File does not exsists! [path={}]".format(path))
                continue
            X_tmp, Y_tmp = self.load_file(path)
            X.append(X_tmp)
            Y.append(Y_tmp)
        self.X = np.concatenate(X, axis=0)
        self.Y = np.concatenate(Y, axis=0)
        logger.info("Success: X={}, Y={}".format(self.X.shape, self.Y.shape))


    def load_file(self, path):
        with h5py.File(path, 'r') as f:
            X = np.array(f["fc"],)
            Y = np.array(f['action/resampled'],)            

        xshape, yshape = X.shape, Y.shape
        X, Y = X.reshape((-1, xshape[-1])), Y.reshape((-1,yshape[-1]))
        return X, Y
        
        
    def __len__(self):
        return len(self.X)
    
    def get_example(self, i):
        return self.X[i], self.Y[i]
