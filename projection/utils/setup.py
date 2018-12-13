import random
import numpy
import chainer

from logging import getLogger, basicConfig, DEBUG, INFO


""" For Reproducibility
"""
def reset_seed(seed=0, *, logger=getLogger(__name__+'.reset_seed')):
    random.seed(seed)
    numpy.random.seed(seed)
    if chainer.cuda.available:
        chainer.cuda.cupy.random.seed(seed)
    logger.info("Reset Seeds ... Done!")
