import os
import sys
import argparse
import numpy as np
import pandas as pd
import time
import threading
import h5py
import glob
from concurrent.futures import ThreadPoolExecutor


from logging import getLogger, basicConfig, DEBUG, INFO
logger = getLogger(__name__)
LOG_FMT = "{asctime} | {levelname:<5s} | {name} | {message}"
basicConfig(level=INFO, format=LOG_FMT, style="{")


# ---------------------------------------------------------------------------------------------------
def resampling(X,th_min=-1., th_max=1.):
    """ 量子化の実行
    
    Args.
    -----
    - x: float
    - th_min/th_max: float, threshhold [unit=degree]
    """
    _X = X.copy()
    _X[X < th_min] = -1.
    _X[X > th_max] =  1.
    _X[(X >= th_min) & (X<= th_max)] = 0
    return _X


# ---------------------------------------------------------------------------------------------------
def threading_clbk(ps):
    (path_in, path_out,) = ps
    
    
    logger.info("Start: Load from {}".format(path_in))
    # Load File
    with h5py.File(path_in, 'r') as f:
        A  = np.array(f["action"],)
        FC = np.array(f["fc"])
    logger.info("Start: A={}, FC={} [from {}]".format(A.shape, FC.shape, path_in))
        
    # 量子化 & Onehot Encoding
    As = resampling(A,)

    shape = list(As.shape) + [3]
    As_onehot = np.eye(3)[As.ravel().astype(int)+1]
    As_onehot  = As_onehot.reshape(shape)
    
    # Write
    with h5py.File(path_out, 'w') as f:
        f.create_dataset("fc", data=FC)
        f.create_group('action')
        f["action"].create_dataset("raw", data=A)
        #f["action"].create_dataset("resampled", data=As)
        f["action"].create_dataset("onehot", data=As_onehot)
    logger.info("Finish: Write to {}".format(path_out))
    return True


# --------------------------------------------------------------------------------------------------
def main(args):

    dir_in  = args.path_in  # "/root/dataStore/tmp2/episodes"
    dir_out = args.path_out # "/root/dataStore/tmp2/Inputs"
    file_list = list(glob.glob(os.path.join(dir_in, "*.h5")))
    file_list.sort()
    file_list = [(path_in, os.path.join(dir_out, path_in.split("/")[-1])) for path_in in file_list]
    logger.info("Target Files: {}".format(len(file_list)))

    # Load files using Threading
    thread_list   = []
    max_worker = 5
    logger.info("Start Load OPP Dataset [{}files]".format(len(file_list)))    
    with ThreadPoolExecutor(max_workers=max_worker) as executor:
        ret = executor.map(threading_clbk, file_list)
    logger.info("Thread ... Finish!! [Results={}]".format(len(list(ret))))
    logger.info("Finish!!")
    

# --------------------------------------------------------------------------------------------------
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-in', required=True,
                       help="path to a input dirctory")
    parser.add_argument('--path-out', required=True,
                       help="path to a output dirctory")
    
    args = parser.parse_args()
    args_dict = vars(args)
    logger.info(" Args:")
    for key in args_dict.keys():
        logger.info(" - {:<15s}= {}".format(key, args_dict[key]))
    print()
    main(args)