import os
import argparse
import numpy as np
import pandas as pd
import h5py
import time

import chainer
import chainer.links as L
import chainer.functions as F
from chainer.training import extensions
from chainer import serializers

from logging import getLogger, basicConfig, DEBUG, INFO
logger = getLogger(__name__)
LOG_FMT = "{asctime} | {levelname:<5s} | {name} | {message}"
basicConfig(level=INFO, format=LOG_FMT, style="{")

from utils.chainer import reset_seed
reset_seed(0)

# -----------------------------------------------------------------------
def make_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='Sub-Commands')
    
    # For TRAINING
    train_parser = subparsers.add_parser('TRAIN')
    train_parser.set_defaults(func=train)
    train_parser.add_argument('-D','--dataset', required=True,
                              help="path to an input data directory")
    train_parser.add_argument('-G','--gpu', default=0, type=int,
                             help="Device identifier. {cpu=-1(default), gpu=0,..}")
    train_parser.add_argument('-B','--batch-size', default=1024, type=int,
                              help="int, Batch size")
    train_parser.add_argument('-E','--epochs', default=1, type=int,
                              help="Int, the number of training epochs")
    train_parser.add_argument('--debug', action='store_true',
                        help="If you want run in debug mode, set this flag.",)
    

    # For TEST
    test_parser = subparsers.add_parser('TEST')
    test_parser.set_defaults(func=test)
    test_parser.add_argument('-D','--dataset', required=True,
                              help="path to an input data directory")
    test_parser.add_argument('-L','--log', required=True,
                              help="path to save log and resutls")    
    test_parser.add_argument('-G','--gpu', default=0, type=int,
                             help="Device identifier. {cpu=-1(default), gpu=0,..}")
    test_parser.add_argument('-B','--batch-size', default=1024, type=int,
                              help="int, Batch size")    
    test_parser.add_argument('--path-model', default="None",
                              help="A path to a saved model, default=None,")
    test_parser.add_argument('--debug', action='store_true',
                             help="If you want run in debug mode, set this flag.",)
    test_parser.add_argument('--data-type', default="TEST",
                             help="Dataset Typem, {TRAIN, VAL, TEST(default)}",)
    return parser

    

# -----------------------------------------------------------------------
def train(args, *, logger=getLogger(__name__+".train")):
    """ Params
    """
    DIR_LOG_ROOT   = ctn.DIR_LOG_BaselineCNN
    DIR_MODEL_ROOT = ctn.DIR_MODEL_BaselineCNN
    MODEL_NAME = "DenseNet"
    
    DS, DS_INFO, DIR_LOG, DIR_MODEL, gpu_id, batch_size, n_epochs = check_params(args, DIR_LOG_ROOT, DIR_MODEL_ROOT,
                                                                                 is_train=True)

    
    """ Load Training & Validation Data
    """
    # Training Data
    logger.info("Load Data ")        
    (X_train, Y_train), dataset_train, iter_train, labels, n_labels = load_dataset(DS, DS_INFO, batch_size, mode="TRAIN")
    (X_val,   Y_val),   dataset_val,   iter_val,   _,      _        = load_dataset(DS, DS_INFO, batch_size, mode="VAL")

    # Info
    logger.info("- Train      : X={}, Y={}".format(X_train.shape, Y_train.shape))
    logger.info("- Validation : X={}, Y={}".format(X_val.shape, Y_val.shape))
    logger.info("- n_labels = {}".format(n_labels))
    logger.info("- labels:")
    for i,key in enumerate(sorted(labels.keys())):
        logger.info("  > {}: {} [{}]".format(i, labels[key]["label"], key))
    logger.info("- dataset_train = {}".format(len(dataset_train)))
    logger.info("- dataset_val   = {}\n".format(len(dataset_val)))
                 
    
    """ Model
    """
    model = ChainerModel(n_in=X_train.shape[-1], n_out=n_labels)
    model = L.Classifier(model, lossfun=F.softmax_cross_entropy)
    model.compute_accuracy = True
    if gpu_id >= 0:
        model.to_gpu(gpu_id)
    logger.info("Success: Building Model")


    """ Trainer & Extention
    """
    # Optimizer
    optimizer = chainer.optimizers.Adam().setup(model)    
    # Update
    updater   = chainer.training.StandardUpdater(iter_train, optimizer,device=gpu_id)
    # Trainer
    trainer   = chainer.training.Trainer(updater, (n_epochs, 'epoch'), out=DIR_LOG)
    trainer.extend(extensions.LogReport())
    logger.info("Success: Build trainer")
    
    """ Extentions
    """
    # trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
    trainer.extend(extensions.Evaluator(iter_val, model, device=gpu_id), name='val')    
    trainer.extend(extensions.PrintReport(['epoch', 'elapsed_time',
                                           'main/loss', 'main/accuracy',
                                           'val/main/loss', 'val/main/accuracy', ]))
    layers = ["conv1", "conv2", "conv3", "fc5","fc6",]
    for l in layers:
        trainer.extend(extensions.ParameterStatistics(eval("model.predictor.{}".format(l)),
                                                      {'std': np.std,'max': np.max,'min':np.min,'mean': np.mean}))        
        trainer.extend(extensions.PlotReport(['{}/W/data/std'.format(l),],
                                             x_key='epoch', file_name='std_{}.png'.format(l)))
        trainer.extend(extensions.PlotReport(['{}/W/data/mean'.format(l),'{}/W/data/max'.format(l), '{}/W/data/min'.format(l)],
                                             x_key='epoch', file_name='range_{}.png'.format(l)))
    trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss',], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'val/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
    trainer.extend(extensions.dump_graph('main/loss'))
    logger.info("Success: Build Extentions\n")
    
    """ Training & Save Model
    """
    logger.info("Start Training")
    trainer.run()
    logger.info("Finish!!\n")
    # Save
    serializers.save_npz(DIR_MODEL, model)
    logger.info("Save model")


    
# -----------------------------------------------------------------------
def test(args, *, logger=getLogger(__name__+".test")):
    """ Params
    """
    DS, DS_INFO, DIR_LOG, DIR_MODEL, gpu_id, batch_size = check_params(args, DIR_LOG_ROOT, DIR_MODEL_ROOT,
                                                                       is_train=False)

    """ Load Training & Validation Data
    """
    logger.info("Load Data ")
    if args.data_type == "TEST":
        mode = "TEST"
    elif args.data_type in ["TRAIN", "VAL"]:
        mode = args.data_type
        logger.warning("Predict on {} (not TEST)".format(mode))
    (X_test,   Y_test), dataset_test, iter_test, labels, n_labels= load_dataset(DS, DS_INFO,  batch_size, mode=mode)
    logger.info("- TEST : X={}, Y={}".format(X_test.shape, Y_test.shape))
    logger.info("- n_labels = {}".format(n_labels))
    logger.info("- labels:")
    for i,key in enumerate(sorted(labels.keys())):
        logger.info("  > {}: {} [{}]".format(i, labels[key]["label"], key))
    logger.info("- dataset_test = {}\n".format(len(dataset_test)))

    
    """ Load Model
    """
    if DS_INFO["task_type"] == "multi_class":
        model = ChainerModel(n_in=X_test.shape[-1], n_out=n_labels)
        model = L.Classifier(model, lossfun=F.softmax_cross_entropy)
        model.compute_accuracy = True
    elif DS_INFO["task_type"] == "multi_task":
        model = ChainerModel(n_in=X_test.shape[-1], n_out=n_labels)
        model = L.Classifier(model, lossfun=F.sigmoid_cross_entropy)
        model.compute_accuracy = False
    else:
        raise ValueError("Invalid task_type")
    chainer.serializers.load_npz(DIR_MODEL, model)    
    if gpu_id >= 0:
        model.to_gpu(gpu_id)
    logger.info("Success: Building Model")


    """ Eval
    """
    # Inference
    from chainer.dataset import concat_examples
    from chainer import cuda
    
    iter_no, y_true, y_pred = 0, [], []
    if   DS_INFO['task_type'] == "multi_class":
        activation_func = F.softmax
    elif DS_INFO['task_type'] == "multi_task":
        activation_func = F.sigmoid
    else:
        pass    

    while True:
        iter_no += 1
        if iter_no%1000 == 0:
            logger.debug("Iteration: {}".format(iter_no))
        test_batch = iter_test.next()
        _x, y_true_tmp = concat_examples(test_batch, device=gpu_id)        
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            y_pred_tmp = activation_func(model.predictor(_x)).data
        y_pred.append(cuda.to_cpu(y_pred_tmp))
        y_true.append(cuda.to_cpu(y_true_tmp))

        if iter_test.is_new_epoch:
            iter_test.reset()
            break
    
    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    logger.debug("- y_true={}, y_pred={}\n".format(y_true.shape, y_pred.shape))

    # Eval
    if   DS_INFO['task_type'] == "multi_class":
        from nn.summary.default import summary_default
        summary_default(y_true, y_pred, labels, DS, DIR_LOG, MODEL_NAME, mode=mode)
    elif DS_INFO['task_type'] == "multi_task":
        from nn.summary.multi_task import summary_multi_task
        summary_multi_task(y_true, y_pred, labels, DS, DIR_LOG, MODEL_NAME, mode=mode)
    else:
        raise ValueError("Invalid task_type")
    
    
    

        
# -----------------------------------------------------------------------
if __name__=='__main__':
    parser = make_parser()
    args = parser.parse_args()
    print()

    
    args_dict = vars(args)
    logger.info(" Args:")
    for key in args_dict.keys():
        logger.info(" - {:<15s}= {}".format(key, args_dict[key]))
    print()
    args.func(args)
