'''
Generic file read, write, pathing module
'''
import os


def smart_path(f):
    '''Auto transforms path into sensible absolute path to minimize usage of path.join(...)'''
    raise NotImplementedError()


def mkdir(f):
    '''Save guard path creation'''
    os.makedirs(f, exist_ok=True)


def pickle(f):
    '''Pickle files'''
    raise NotImplementedError()


def read(f):
    '''Generic read method with auto data type detection'''
    raise NotImplementedError()


def write(f):
    '''Generic write method with auto data type detection'''
    raise NotImplementedError()
