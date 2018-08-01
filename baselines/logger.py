import os
import sys
import shutil
import json
import time
import datetime
import tempfile
from collections import defaultdict

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40

DISABLED = 50


class KVWriter(object):
    """
    Key Value writer
    """
    def writekvs(self, kvs):
        """
        write a dictionary to file

        :param kvs: (dict)
        """
        raise NotImplementedError


class SeqWriter(object):
    """
    sequence writer
    """
    def writeseq(self, seq):
        """
        write an array to file

        :param seq: (list)
        """
        raise NotImplementedError


class HumanOutputFormat(KVWriter, SeqWriter):
    def __init__(self, filename_or_file):
        """
        log to a file, in a human readable format

        :param filename_or_file: (str or File) the file to write the log to
        """
        if isinstance(filename_or_file, str):
            self.file = open(filename_or_file, 'wt')
            self.own_file = True
        else:
            assert hasattr(filename_or_file, 'read'), 'expected file or str, got %s' % filename_or_file
            self.file = filename_or_file
            self.own_file = False

    def writekvs(self, kvs):
        # Create strings for printing
        key2str = {}
        for (key, val) in sorted(kvs.items()):
            if isinstance(val, float):
                valstr = '%-8.3g' % (val,)
            else:
                valstr = str(val)
            key2str[self._truncate(key)] = self._truncate(valstr)

        # Find max widths
        if len(key2str) == 0:
            print('WARNING: tried to write empty key-value dict')
            return
        else:
            keywidth = max(map(len, key2str.keys()))
            valwidth = max(map(len, key2str.values()))

        # Write out the data
        dashes = '-' * (keywidth + valwidth + 7)
        lines = [dashes]
        for (key, val) in sorted(key2str.items()):
            lines.append('| %s%s | %s%s |' % (
                key,
                ' ' * (keywidth - len(key)),
                val,
                ' ' * (valwidth - len(val)),
            ))
        lines.append(dashes)
        self.file.write('\n'.join(lines) + '\n')

        # Flush the output to the file
        self.file.flush()

    @classmethod
    def _truncate(cls, string):
        return string[:20] + '...' if len(string) > 23 else string

    def writeseq(self, seq):
        seq = list(seq)
        for (i, elem) in enumerate(seq):
            self.file.write(elem)
            if i < len(seq) - 1:  # add space unless this is the last one
                self.file.write(' ')
        self.file.write('\n')
        self.file.flush()

    def close(self):
        """
        closes the file
        """
        if self.own_file:
            self.file.close()


class JSONOutputFormat(KVWriter):
    def __init__(self, filename):
        """
        log to a file, in the JSON format

        :param filename: (str) the file to write the log to
        """
        self.file = open(filename, 'wt')

    def writekvs(self, kvs):
        for key, value in sorted(kvs.items()):
            if hasattr(value, 'dtype'):
                value = value.tolist()
                kvs[key] = float(value)
        self.file.write(json.dumps(kvs) + '\n')
        self.file.flush()

    def close(self):
        """
        closes the file
        """
        self.file.close()


class CSVOutputFormat(KVWriter):
    def __init__(self, filename):
        """
        log to a file, in a CSV format

        :param filename: (str) the file to write the log to
        """
        self.file = open(filename, 'w+t')
        self.keys = []
        self.sep = ','

    def writekvs(self, kvs):
        # Add our current row to the history
        extra_keys = kvs.keys() - self.keys
        if extra_keys:
            self.keys.extend(extra_keys)
            self.file.seek(0)
            lines = self.file.readlines()
            self.file.seek(0)
            for (i, key) in enumerate(self.keys):
                if i > 0:
                    self.file.write(',')
                self.file.write(key)
            self.file.write('\n')
            for line in lines[1:]:
                self.file.write(line[:-1])
                self.file.write(self.sep * len(extra_keys))
                self.file.write('\n')
        for i, key in enumerate(self.keys):
            if i > 0:
                self.file.write(',')
            value = kvs.get(key)
            if value is not None:
                self.file.write(str(value))
        self.file.write('\n')
        self.file.flush()

    def close(self):
        """
        closes the file
        """
        self.file.close()


class TensorBoardOutputFormat(KVWriter):
    def __init__(self, folder):
        """
        Dumps key/value pairs into TensorBoard's numeric format.

        :param folder: (str) the folder to write the log to
        """
        os.makedirs(folder, exist_ok=True)
        self.dir = folder
        self.step = 1
        prefix = 'events'
        path = os.path.join(os.path.abspath(folder), prefix)
        import tensorflow as tf
        from tensorflow.python import pywrap_tensorflow
        from tensorflow.core.util import event_pb2
        from tensorflow.python.util import compat
        self._tf = tf
        self.event_pb2 = event_pb2
        self.pywrap_tensorflow = pywrap_tensorflow
        self.writer = pywrap_tensorflow.EventsWriter(compat.as_bytes(path))

    def writekvs(self, kvs):
        def summary_val(key, value):
            kwargs = {'tag': key, 'simple_value': float(value)}
            return self._tf.Summary.Value(**kwargs)

        summary = self._tf.Summary(value=[summary_val(k, v) for k, v in kvs.items()])
        event = self.event_pb2.Event(wall_time=time.time(), summary=summary)
        event.step = self.step  # is there any reason why you'd want to specify the step?
        self.writer.WriteEvent(event)
        self.writer.Flush()
        self.step += 1

    def close(self):
        """
        closes the file
        """
        if self.writer:
            self.writer.Close()
            self.writer = None


def make_output_format(_format, ev_dir, log_suffix=''):
    """
    return a logger for the requested format

    :param _format: (str) the requested format to log to ('stdout', 'log', 'json', 'csv' or 'tensorboard')
    :param ev_dir: (str) the logging directory
    :param log_suffix: (str) the suffix for the log file
    :return: (KVWrite) the logger
    """
    os.makedirs(ev_dir, exist_ok=True)
    if _format == 'stdout':
        return HumanOutputFormat(sys.stdout)
    elif _format == 'log':
        return HumanOutputFormat(os.path.join(ev_dir, 'log%s.txt' % log_suffix))
    elif _format == 'json':
        return JSONOutputFormat(os.path.join(ev_dir, 'progress%s.json' % log_suffix))
    elif _format == 'csv':
        return CSVOutputFormat(os.path.join(ev_dir, 'progress%s.csv' % log_suffix))
    elif _format == 'tensorboard':
        return TensorBoardOutputFormat(os.path.join(ev_dir, 'tb%s' % log_suffix))
    else:
        raise ValueError('Unknown format specified: %s' % (_format,))


# ================================================================
# API
# ================================================================

def logkv(key, val):
    """
    Log a value of some diagnostic
    Call this once for each diagnostic quantity, each iteration
    If called many times, last value will be used.

    :param key: (Any) save to log this key
    :param val: (Any) save to log this value
    """
    Logger.CURRENT.logkv(key, val)


def logkv_mean(key, val):
    """
    The same as logkv(), but if called many times, values averaged.

    :param key: (Any) save to log this key
    :param val: (Number) save to log this value
    """
    Logger.CURRENT.logkv_mean(key, val)


def logkvs(key_values):
    """
    Log a dictionary of key-value pairs

    :param key_values: (dict) the list of keys and values to save to log
    """
    for key, value in key_values.items():
        logkv(key, value)


def dumpkvs():
    """
    Write all of the diagnostics from the current iteration
    """
    Logger.CURRENT.dumpkvs()


def getkvs():
    """
    get the key values logs

    :return: (dict) the logged values
    """
    return Logger.CURRENT.name2val


def log(*args, level=INFO):
    """
    Write the sequence of args, with no separators,
    to the console and output files (if you've configured an output file).

    level: int. (see logger.py docs) If the global logger level is higher than
                the level argument here, don't print to stdout.

    :param args: (list) log the arguments
    :param level: (int) the logging level (can be DEBUG=10, INFO=20, WARN=30, ERROR=40, DISABLED=50)
    """
    Logger.CURRENT.log(*args, level=level)


def debug(*args):
    """
    Write the sequence of args, with no separators,
    to the console and output files (if you've configured an output file).
    Using the DEBUG level.

    :param args: (list) log the arguments
    """
    log(*args, level=DEBUG)


def info(*args):
    """
    Write the sequence of args, with no separators,
    to the console and output files (if you've configured an output file).
    Using the INFO level.

    :param args: (list) log the arguments
    """
    log(*args, level=INFO)


def warn(*args):
    """
    Write the sequence of args, with no separators,
    to the console and output files (if you've configured an output file).
    Using the WARN level.

    :param args: (list) log the arguments
    """
    log(*args, level=WARN)


def error(*args):
    """
    Write the sequence of args, with no separators,
    to the console and output files (if you've configured an output file).
    Using the ERROR level.

    :param args: (list) log the arguments
    """
    log(*args, level=ERROR)


def set_level(level):
    """
    Set logging threshold on current logger.

    :param level: (int) the logging level (can be DEBUG=10, INFO=20, WARN=30, ERROR=40, DISABLED=50)
    """
    Logger.CURRENT.set_level(level)


def get_level():
    """
    Get logging threshold on current logger.
    :return: (int) the logging level (can be DEBUG=10, INFO=20, WARN=30, ERROR=40, DISABLED=50)
    """
    return Logger.CURRENT.level


def get_dir():
    """
    Get directory that log files are being written to.
    will be None if there is no output directory (i.e., if you didn't call start)

    :return: (str) the logging directory
    """
    return Logger.CURRENT.get_dir()


record_tabular = logkv
dump_tabular = dumpkvs


class ProfileKV:
    def __init__(self, name):
        """
        Usage:
        with logger.ProfileKV("interesting_scope"):
            code

        :param name: (str) the profiling name
        """
        self.name = "wait_" + name

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, _type, value, traceback):
        Logger.CURRENT.name2val[self.name] += time.time() - self.start_time


def profile(name):
    """
    Usage:
    @profile("my_func")
    def my_func(): code

    :param name: (str) the profiling name
    :return: (function) the wrapped function
    """
    def decorator_with_name(func):
        def func_wrapper(*args, **kwargs):
            with ProfileKV(name):
                return func(*args, **kwargs)

        return func_wrapper

    return decorator_with_name


# ================================================================
# Backend
# ================================================================

class Logger(object):
    # A logger with no output files. (See right below class definition)
    #  So that you can still log to the terminal without setting up any output files
    DEFAULT = None
    CURRENT = None  # Current logger being used by the free functions above

    def __init__(self, folder, output_formats):
        """
        the logger class

        :param folder: (str) the logging location
        :param output_formats: ([str]) the list of output format
        """
        self.name2val = defaultdict(float)  # values this iteration
        self.name2cnt = defaultdict(int)
        self.level = INFO
        self.dir = folder
        self.output_formats = output_formats

    # Logging API, forwarded
    # ----------------------------------------
    def logkv(self, key, val):
        """
        Log a value of some diagnostic
        Call this once for each diagnostic quantity, each iteration
        If called many times, last value will be used.

        :param key: (Any) save to log this key
        :param val: (Any) save to log this value
        """
        self.name2val[key] = val

    def logkv_mean(self, key, val):
        """
        The same as logkv(), but if called many times, values averaged.

        :param key: (Any) save to log this key
        :param val: (Number) save to log this value
        """
        if val is None:
            self.name2val[key] = None
            return
        oldval, cnt = self.name2val[key], self.name2cnt[key]
        self.name2val[key] = oldval * cnt / (cnt + 1) + val / (cnt + 1)
        self.name2cnt[key] = cnt + 1

    def dumpkvs(self):
        """
        Write all of the diagnostics from the current iteration
        """
        if self.level == DISABLED:
            return
        for fmt in self.output_formats:
            if isinstance(fmt, KVWriter):
                fmt.writekvs(self.name2val)
        self.name2val.clear()
        self.name2cnt.clear()

    def log(self, *args, level=INFO):
        """
        Write the sequence of args, with no separators,
        to the console and output files (if you've configured an output file).

        level: int. (see logger.py docs) If the global logger level is higher than
                    the level argument here, don't print to stdout.

        :param args: (list) log the arguments
        :param level: (int) the logging level (can be DEBUG=10, INFO=20, WARN=30, ERROR=40, DISABLED=50)
        """
        if self.level <= level:
            self._do_log(args)

    # Configuration
    # ----------------------------------------
    def set_level(self, level):
        """
        Set logging threshold on current logger.

        :param level: (int) the logging level (can be DEBUG=10, INFO=20, WARN=30, ERROR=40, DISABLED=50)
        """
        self.level = level

    def get_dir(self):
        """
        Get directory that log files are being written to.
        will be None if there is no output directory (i.e., if you didn't call start)

        :return: (str) the logging directory
        """
        return self.dir

    def close(self):
        """
        closes the file
        """
        for fmt in self.output_formats:
            fmt.close()

    # Misc
    # ----------------------------------------
    def _do_log(self, args):
        """
        log to the requested format outputs

        :param args: (list) the arguments to log
        """
        for fmt in self.output_formats:
            if isinstance(fmt, SeqWriter):
                fmt.writeseq(map(str, args))


Logger.DEFAULT = Logger.CURRENT = Logger(folder=None, output_formats=[HumanOutputFormat(sys.stdout)])


def configure(folder=None, format_strs=None):
    """
    configure the current logger

    :param folder: (str) the save location (if None, $OPENAI_LOGDIR, if still None, tempdir/openai-[date & time])
    :param format_strs: (list) the output logging format
        (if None, $OPENAI_LOG_FORMAT, if still None, ['stdout', 'log', 'csv'])
    """
    if folder is None:
        folder = os.getenv('OPENAI_LOGDIR')
    if folder is None:
        folder = os.path.join(tempfile.gettempdir(), datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"))
    assert isinstance(folder, str)
    os.makedirs(folder, exist_ok=True)

    log_suffix = ''
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    if rank > 0:
        log_suffix = "-rank%03i" % rank

    if format_strs is None:
        if rank == 0:
            format_strs = os.getenv('OPENAI_LOG_FORMAT', 'stdout,log,csv').split(',')
        else:
            format_strs = os.getenv('OPENAI_LOG_FORMAT_MPI', 'log').split(',')
    format_strs = filter(None, format_strs)
    output_formats = [make_output_format(f, folder, log_suffix) for f in format_strs]

    Logger.CURRENT = Logger(folder=folder, output_formats=output_formats)
    log('Logging to %s' % folder)


def reset():
    """
    reset the current logger
    """
    if Logger.CURRENT is not Logger.DEFAULT:
        Logger.CURRENT.close()
        Logger.CURRENT = Logger.DEFAULT
        log('Reset logger')


class ScopedConfigure(object):
    def __init__(self, folder=None, format_strs=None):
        """
        Class for using context manager while logging

        usage:
        with ScopedConfigure(folder=None, format_strs=None):
            {code}

        :param folder: (str) the logging folder
        :param format_strs: ([str]) the list of output logging format
        """
        self.dir = folder
        self.format_strs = format_strs
        self.prevlogger = None

    def __enter__(self):
        self.prevlogger = Logger.CURRENT
        configure(folder=self.dir, format_strs=self.format_strs)

    def __exit__(self, *args):
        Logger.CURRENT.close()
        Logger.CURRENT = self.prevlogger


# ================================================================

def _demo():
    """
    tests for the logger module
    """
    info("hi")
    debug("shouldn't appear")
    set_level(DEBUG)
    debug("should appear")
    folder = "/tmp/testlogging"
    if os.path.exists(folder):
        shutil.rmtree(folder)
    configure(folder=folder)
    logkv("a", 3)
    logkv("b", 2.5)
    dumpkvs()
    logkv("b", -2.5)
    logkv("a", 5.5)
    dumpkvs()
    info("^^^ should see a = 5.5")
    logkv_mean("b", -22.5)
    logkv_mean("b", -44.4)
    logkv("a", 5.5)
    dumpkvs()
    with ScopedConfigure(None, None):
        info("^^^ should see b = 33.3")

    with ScopedConfigure("/tmp/test-logger/", ["json"]):
        logkv("b", -2.5)
        dumpkvs()

    reset()
    logkv("a", "longasslongasslongasslongasslongasslongassvalue")
    dumpkvs()
    warn("hey")
    error("oh")
    logkvs({"test": 1})


# ================================================================
# Readers
# ================================================================

def read_json(fname):
    """
    read a json file using pandas

    :param fname: (str) the file path to read
    :return: (pandas DataFrame) the data in the json
    """
    import pandas
    data = []
    with open(fname, 'rt') as file_handler:
        for line in file_handler:
            data.append(json.loads(line))
    return pandas.DataFrame(data)


def read_csv(fname):
    """
    read a csv file using pandas

    :param fname: (str) the file path to read
    :return: (pandas DataFrame) the data in the csv
    """
    import pandas
    return pandas.read_csv(fname, index_col=None, comment='#')


def read_tb(path):
    """
    read a tensorboard output

    :param path: (str) a tensorboard file OR a directory, where we will find all TB files of the form events.
    :return: (pandas DataFrame) the tensorboad data
    """
    import pandas
    import numpy as np
    from glob import glob
    # from collections import defaultdict
    import tensorflow as tf
    if os.path.isdir(path):
        fnames = glob(os.path.join(path, "events.*"))
    elif os.path.basename(path).startswith("events."):
        fnames = [path]
    else:
        raise NotImplementedError("Expected tensorboard file or directory containing them. Got %s" % path)
    tag2pairs = defaultdict(list)
    maxstep = 0
    for fname in fnames:
        for summary in tf.train.summary_iterator(fname):
            if summary.step > 0:
                for value in summary.summary.value:
                    pair = (summary.step, value.simple_value)
                    tag2pairs[value.tag].append(pair)
                maxstep = max(summary.step, maxstep)
    data = np.empty((maxstep, len(tag2pairs)))
    data[:] = np.nan
    tags = sorted(tag2pairs.keys())
    for (colidx, tag) in enumerate(tags):
        pairs = tag2pairs[tag]
        for (step, value) in pairs:
            data[step - 1, colidx] = value
    return pandas.DataFrame(data, columns=tags)


if __name__ == "__main__":
    _demo()
