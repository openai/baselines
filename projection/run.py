import os
import argparse
import numpy as np
import pandas as pd
import h5py
import time
import glob

import chainer
import chainer.links as L
import chainer.functions as F
from chainer.training import extensions
from chainer import serializers

from logging import getLogger, basicConfig, DEBUG, INFO
logger = getLogger(__name__)
LOG_FMT = "{asctime} | {levelname:<5s} | {name} | {message}"
basicConfig(level=INFO, format=LOG_FMT, style="{")

from utils.setup import reset_seed
reset_seed(0)


from dataset.default import DefaultDataset
from models.dense    import DenseNet

# -----------------------------------------------------------------------
def make_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='Sub-Commands')
    
    # For TRAINING
    train_parser = subparsers.add_parser('TRAIN')
    train_parser.set_defaults(func=train)
    train_parser.add_argument('--path-data-train', required=True,
                              help="path to an training data directory")
    train_parser.add_argument('--path-data-val', required=True,
                              help="path to an test data directory")
    train_parser.add_argument('--path-model', required=True,
                              help="path to save the trained model")
    train_parser.add_argument('--path-log', required=True,
                              help="path to save log data.")
    train_parser.add_argument('-G','--gpu', default=0, type=int,
                             help="Device identifier. {cpu=-1(default), gpu=0,..}")
    train_parser.add_argument('-B','--batch-size', default=1024, type=int,
                              help="int, Batch size")
    train_parser.add_argument('-E','--epochs', default=1, type=int,
                              help="Int, the number of training epochs")
    # train_parser.add_argument('--debug', action='store_true',
    #                           help="If you want run in debug mode, set this flag.",)
    

    # For TEST
    test_parser = subparsers.add_parser('TEST')
    test_parser.set_defaults(func=test)
    test_parser.add_argument('--path-data-test', required=True,
                              help="path to an test data directory")
    test_parser.add_argument('--path-model', required=True,
                              help="path to save the trained model")    
    test_parser.add_argument('--path-log', required=True,
                              help="path to save log and resutls")
    test_parser.add_argument('-G','--gpu', default=0, type=int,
                             help="Device identifier. {cpu=-1(default), gpu=0,..}")
    test_parser.add_argument('-B','--batch-size', default=1024, type=int,
                              help="int, Batch size")    
    test_parser.add_argument('--debug', action='store_true',
                             help="If you want run in debug mode, set this flag.",)
    # test_parser.add_argument('--data-type', default="TEST",
    #                          help="Dataset Typem, {TRAIN, VAL, TEST(default)}",)
    return parser

    

# -----------------------------------------------------------------------
def train(args, *, logger=getLogger(__name__+".train")):
    """ Params
    """
    MODEL_NAME     = "DenseNet"
    DIR_DATA_TRAIN = args.path_data_train
    DIR_DATA_VAL   = args.path_data_val
    DIR_LOG        = args.path_log
    DIR_MODEL      = args.path_model
    gpu_id         = int(args.gpu)
    batch_size     = int(args.batch_size)
    n_epochs       = int(args.epochs)

    
    """ Load Training & Validation Data
    """
    # Training Data
    logger.info("Load Dataset")
    ## Dataset
    file_list_train = list(glob.glob(os.path.join(DIR_DATA_TRAIN, "*.h5")))
    file_list_train.sort()
    file_list_val = list(glob.glob(os.path.join(DIR_DATA_VAL, "*.h5")))
    file_list_val.sort()
    logger.info("- Train: {} files".format(len(file_list_train)))
    logger.info("- Val: {} files".format(len(file_list_val)))    
    dataset_train = DefaultDataset(file_list_train)
    dataset_val   = DefaultDataset(file_list_val)

    # Iterator
    iter_train = chainer.iterators.SerialIterator(dataset_train, batch_size, repeat=True, shuffle=True)
    iter_val   = chainer.iterators.SerialIterator(dataset_val,   batch_size, repeat=False, shuffle=False)    

    # Info
    logger.info("- Train      : X={}, Y={}".format(dataset_train.X.shape, dataset_train.Y.shape))
    logger.info("- Validation : X={}, Y={}".format(dataset_val.X.shape, dataset_val.Y.shape))
    
    
    """ Model
    """
    model = DenseNet(n_in=dataset_train.X.shape[-1], n_out=dataset_train.Y.shape[-1])
    model = L.Classifier(model, lossfun=F.mean_squared_error)
    model.compute_accuracy = False
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
    trainer.extend(extensions.Evaluator(iter_val, model, device=gpu_id), name='val')    
    trainer.extend(extensions.PrintReport(['epoch', 'elapsed_time',
                                           'main/loss', 'main/accuracy',
                                           'val/main/loss', 'val/main/accuracy', ]))
    layers = ["fc1","fc2","fc3",]
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
    MODEL_NAME     = "DenseNet"
    DIR_DATA_TEST  = args.path_data_test
    DIR_LOG        = args.path_log
    DIR_MODEL      = args.path_model
    gpu_id         = int(args.gpu)
    batch_size     = int(args.batch_size)
    

    """ Load Training & Validation Data
    """
    logger.info("Load Data ")
    ## Dataset
    file_list_test = list(glob.glob(os.path.join(DIR_DATA_TEST, "*.h5")))
    file_list_test.sort()
    logger.info("- Test: {} files".format(len(file_list_test)))
    dataset_test   = DefaultDataset(file_list_test)

    # Iterator
    iter_test = chainer.iterators.SerialIterator(dataset_test,   batch_size, repeat=False, shuffle=False)    

    # Info
    logger.info("- Test      : X={}, Y={}".format(dataset_test.X.shape, dataset_test.Y.shape))


    
    """ Load Model
    """
    model = DenseNet(n_in=dataset_test.X.shape[-1], n_out=dataset_test.Y.shape[-1])
    model = L.Classifier(model, lossfun=F.mean_squared_error)
    model.compute_accuracy = True
    chainer.serializers.load_npz(DIR_MODEL, model)    
    if gpu_id >= 0:
        model.to_gpu(gpu_id)
    logger.info("Success: Building Model")


    """ Eval
    """
    # Inference
    from chainer.dataset import concat_examples
    from chainer import cuda
    iter_no, x_in, y_true, y_pred = 0, [], [], []
    y_inter = {}

    while True:
        iter_no += 1
        if iter_no%1000 == 0:
            logger.debug("Iteration: {}".format(iter_no))
        test_batch = iter_test.next()
        _x, y_true_tmp = concat_examples(test_batch, device=gpu_id)
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            # y_pred_tmp = model.predictor(_x).data
            y_pred_tmp, y_inter_tmp = model.get_inter_layer(_x)
            
        x_in.append(cuda.to_cpu(_x))
        y_pred.append(cuda.to_cpu(y_pred_tmp))
        y_true.append(cuda.to_cpu(y_true_tmp))
        
        for _y in y_inter_tmp:
            if len(y_inter) == 0:
                for key in y_inter_tmp.keys():                    
                    y_inter[key] = [cuda.to_cpu(y_inter_tmp[key]),]
            else:
                for key in y_inter_tmp.keys():                    
                    y_inter[key].append([cuda.to_cpu(y_inter_tmp[key]),])
                    

        if iter_test.is_new_epoch:
            iter_test.reset()
            break

        
    x_in   = np.concatenate(x_in,   axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    logger.debug("- x_in, y_true={}, y_pred={}\n".format(x_in.shape, y_true.shape, y_pred.shape))
    for key in y_inter.keys():
        y_inter[key] = np.concatenate(y_inter[key], axis=0)
    
    
    # Eval
    df_in   = pd.DataFrame(x_in)
    df_in.columns = ["x_{}".format(c) for c in list(df_in.columns)]
    df_pred = pd.DataFrame(y_pred)
    df_pred.columns = ["pred_{}".format(c) for c in list(df_pred.columns)]
    df_true = pd.DataFrame(y_true)
    df_true.columns = ["true_{}".format(c) for c in list(df_true.columns)]
    df_pred = pd.concat([df_in, df_pred,df_true], axis=1)
    logger.info("df_pred = \n {}".format(df_pred.head()))
    filename = os.path.join(DIR_LOG, "pred_detail.csv")
    df_pred.to_csv(filename)
    logger.info("Write results to {} [df_pred={}]".format(filename, df_pred.shape))


    # Save Internal State
    filename = os.path.join(DIR_LOG, "pred_inernal.csv")
    with h5py.File(filename, "w") as f:
        # Intermidiate output after applying activation functions
        f.create_group('post_act')
        for key in y_inter.keys():
            f["post_act"].create_dataset(key, data=y_inter[key])
            

            
    
    # Summary
    ## MSE
    mse = np.mean((y_pred - y_true)**2)
    mae = np.mean(np.absolute(y_pred - y_true))
    logger.info("=== Summary ===")
    logger.info("MSE: {}".format(mse))
    logger.info("MAE: {}".format(mae))
    logger.info("===============")
    
    
    

        
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
