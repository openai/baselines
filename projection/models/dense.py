import chainer
import chainer.links as L
import chainer.functions as F

import numpy as np
import cupy as cp



class DenseNet(chainer.Chain):
    """
    Reference.
    ----------
    - "Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition"
    [www.mdpi.com/1424-8220/16/1/115/pdf]
    - Baseline CNN
    """
    """ 
    Args.
    -----
    - n_in  : int, Input  dim (=X.shape[-1])
    - n_out : int, Output dim (=Y.shape[-1])
    """
    def __init__(self, n_in=None, n_out=None):
        super(DenseNet, self,).__init__()        
        with self.init_scope():            
            # FC
            self.fc1   = L.Linear(n_in, 32)
            self.fc2   = L.Linear(32,   32)
            self.fc3   = L.Linear(32,   n_out)            
            
    def __call__(self, x):
        # Full Connected
        h1 = F.dropout(F.relu(self.fc1(x)))
        h2 = F.dropout(F.relu(self.fc2(h1)))
        h3 = F.tanh(self.fc3(h2))
        return h3
