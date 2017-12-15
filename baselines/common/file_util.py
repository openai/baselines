'''
Generic file read, write, pathing module
'''
import cloudpickle
import os
import pickle


def smart_path(filepath):
    '''Auto transforms path into sensible absolute path to minimize usage of path.join(...)'''
    raise NotImplementedError()


def mkdir(filepath):
    '''Save guard path creation'''
    os.makedirs(filepath, exist_ok=True)


def pickle_fn(filepath, fn):
    '''Pickle a function'''
    byte = cloudpickle.dumps(fn)
    with open(filepath, 'wb') as f:
        f.write(byte)
    return fn


def unpickle_fn(filepath):
    '''Un-pickle a function'''
    byte = open(filepath, 'rb').read()
    fn = pickle.loads(byte)
    return fn


def read(filepath):
    '''Generic read method with auto data type detection'''
    raise NotImplementedError()


def write(filepath):
    '''Generic write method with auto data type detection'''
    raise NotImplementedError()
