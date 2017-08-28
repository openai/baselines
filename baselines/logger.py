import os
import sys
import shutil
import os.path as osp
import json
import time
import datetime
import tempfile

LOG_OUTPUT_FORMATS = ['stdout', 'log', 'json']

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40

DISABLED = 50

class OutputFormat(object):
    def writekvs(self, kvs):
        """
        Write key-value pairs
        """
        raise NotImplementedError

    def writeseq(self, args):
        """
        Write a sequence of other data (e.g. a logging message)
        """
        pass

    def close(self):
        return


class HumanOutputFormat(OutputFormat):
    def __init__(self, file):
        self.file = file

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

    def _truncate(self, s):
        return s[:20] + '...' if len(s) > 23 else s

    def writeseq(self, args):
        for arg in args:
            self.file.write(arg)
        self.file.write('\n')
        self.file.flush()

class JSONOutputFormat(OutputFormat):
    def __init__(self, file):
        self.file = file

    def writekvs(self, kvs):
        for k, v in sorted(kvs.items()):
            if hasattr(v, 'dtype'):
                v = v.tolist()
                kvs[k] = float(v)
        self.file.write(json.dumps(kvs) + '\n')
        self.file.flush()

class TensorBoardOutputFormat(OutputFormat):
    """
    Dumps key/value pairs into TensorBoard's numeric format.
    """
    def __init__(self, dir):
        os.makedirs(dir, exist_ok=True)
        self.dir = dir
        self.step = 1
        prefix = 'events'
        path = osp.join(osp.abspath(dir), prefix)
        import tensorflow as tf
        from tensorflow.python import pywrap_tensorflow        
        from tensorflow.core.util import event_pb2
        from tensorflow.python.util import compat
        self.tf = tf
        self.event_pb2 = event_pb2
        self.pywrap_tensorflow = pywrap_tensorflow
        self.writer = pywrap_tensorflow.EventsWriter(compat.as_bytes(path))

    def writekvs(self, kvs):
        def summary_val(k, v):
            kwargs = {'tag': k, 'simple_value': float(v)}
            return self.tf.Summary.Value(**kwargs)
        summary = self.tf.Summary(value=[summary_val(k, v) for k, v in kvs.items()])
        event = self.event_pb2.Event(wall_time=time.time(), summary=summary)
        event.step = self.step # is there any reason why you'd want to specify the step?
        self.writer.WriteEvent(event)
        self.writer.Flush()
        self.step += 1

    def close(self):
        if self.writer:
            self.writer.Close()
            self.writer = None


def make_output_format(format, ev_dir):
    os.makedirs(ev_dir, exist_ok=True)
    if format == 'stdout':
        return HumanOutputFormat(sys.stdout)
    elif format == 'log':
        log_file = open(osp.join(ev_dir, 'log.txt'), 'wt')
        return HumanOutputFormat(log_file)
    elif format == 'json':
        json_file = open(osp.join(ev_dir, 'progress.json'), 'wt')
        return JSONOutputFormat(json_file)
    elif format == 'tensorboard':
        return TensorBoardOutputFormat(osp.join(ev_dir, 'tb'))
    else:
        raise ValueError('Unknown format specified: %s' % (format,))

# ================================================================
# API
# ================================================================

def logkv(key, val):
    """
    Log a value of some diagnostic
    Call this once for each diagnostic quantity, each iteration
    """
    Logger.CURRENT.logkv(key, val)

def logkvs(d):
    """
    Log a dictionary of key-value pairs
    """
    for (k, v) in d.items():
        logkv(k, v)

def dumpkvs():
    """
    Write all of the diagnostics from the current iteration

    level: int. (see logger.py docs) If the global logger level is higher than
                the level argument here, don't print to stdout.
    """
    Logger.CURRENT.dumpkvs()

def getkvs():
    return Logger.CURRENT.name2val    


def log(*args, level=INFO):
    """
    Write the sequence of args, with no separators, to the console and output files (if you've configured an output file).
    """
    Logger.CURRENT.log(*args, level=level)


def debug(*args):
    log(*args, level=DEBUG)


def info(*args):
    log(*args, level=INFO)


def warn(*args):
    log(*args, level=WARN)


def error(*args):
    log(*args, level=ERROR)


def set_level(level):
    """
    Set logging threshold on current logger.
    """
    Logger.CURRENT.set_level(level)

def get_dir():
    """
    Get directory that log files are being written to.
    will be None if there is no output directory (i.e., if you didn't call start)
    """
    return Logger.CURRENT.get_dir()

record_tabular = logkv
dump_tabular = dumpkvs

# ================================================================
# Backend
# ================================================================

class Logger(object):
    DEFAULT = None  # A logger with no output files. (See right below class definition)
                    # So that you can still log to the terminal without setting up any output files
    CURRENT = None  # Current logger being used by the free functions above

    def __init__(self, dir, output_formats):
        self.name2val = {}  # values this iteration
        self.level = INFO
        self.dir = dir
        self.output_formats = output_formats

    # Logging API, forwarded
    # ----------------------------------------
    def logkv(self, key, val):
        self.name2val[key] = val

    def dumpkvs(self):
        if self.level == DISABLED: return
        for fmt in self.output_formats:
            fmt.writekvs(self.name2val)
        self.name2val.clear()

    def log(self, *args, level=INFO):
        if self.level <= level:
            self._do_log(args)

    # Configuration
    # ----------------------------------------
    def set_level(self, level):
        self.level = level

    def get_dir(self):
        return self.dir

    def close(self):
        for fmt in self.output_formats:
            fmt.close()

    # Misc
    # ----------------------------------------
    def _do_log(self, args):
        for fmt in self.output_formats:
            fmt.writeseq(args)

Logger.DEFAULT = Logger.CURRENT = Logger(dir=None, output_formats=[HumanOutputFormat(sys.stdout)])

def configure(dir=None, format_strs=None):
    assert Logger.CURRENT is Logger.DEFAULT,\
        "Only call logger.configure() when it's in the default state. Try calling logger.reset() first."
    prevlogger = Logger.CURRENT
    if dir is None:
        dir = os.getenv('OPENAI_LOGDIR')
    if dir is None:
        dir = osp.join(tempfile.gettempdir(), 
            datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"))
    if format_strs is None:
        format_strs = LOG_OUTPUT_FORMATS
    output_formats = [make_output_format(f, dir) for f in format_strs]
    Logger.CURRENT = Logger(dir=dir, output_formats=output_formats)
    log('Logging to %s'%dir)

if os.getenv('OPENAI_LOGDIR'): 
    # if OPENAI_LOGDIR is set, configure the logger on import
    # this kind of nasty (unexpected to user), but I don't know how else to inject the logger
    # to a script that's getting run in a subprocess
    configure(dir=os.getenv('OPENAI_LOGDIR'))

def reset():
    Logger.CURRENT = Logger.DEFAULT
    log('Reset logger')

# ================================================================

def _demo():
    info("hi")
    debug("shouldn't appear")
    set_level(DEBUG)
    debug("should appear")
    dir = "/tmp/testlogging"
    if os.path.exists(dir):
        shutil.rmtree(dir)
    with session(dir=dir):
        logkv("a", 3)
        logkv("b", 2.5)
        dumpkvs()
        logkv("b", -2.5)
        logkv("a", 5.5)
        dumpkvs()
        info("^^^ should see a = 5.5")

    logkv("b", -2.5)
    dumpkvs()

    logkv("a", "longasslongasslongasslongasslongasslongassvalue")
    dumpkvs()


if __name__ == "__main__":
    _demo()
