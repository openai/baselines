from __future__ import print_function
from contextlib import contextmanager
import numpy as np
import time
import shlex
import subprocess

# ================================================================
# Misc
# ================================================================

def fmt_row(width, row, header=False):
    out = " | ".join(fmt_item(x, width) for x in row)
    if header: out = out + "\n" + "-"*len(out)
    return out

def fmt_item(x, l):
    if isinstance(x, np.ndarray):
        assert x.ndim==0
        x = x.item()
    if isinstance(x, (float, np.float32, np.float64)):
        v = abs(x)
        if (v < 1e-4 or v > 1e+4) and v > 0:
            rep = "%7.2e" % x
        else:
            rep = "%7.5f" % x
    else: rep = str(x)
    return " "*(l - len(rep)) + rep

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

def colorize(string, color='green', bold=False, highlight=False):
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

def print_cmd(cmd, dry=False):
    if isinstance(cmd, str):  # for shell=True
        pass
    else:
        cmd = ' '.join(shlex.quote(arg) for arg in cmd)
    print(colorize(('CMD: ' if not dry else 'DRY: ') + cmd))


def get_git_commit(cwd=None):
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=cwd).decode('utf8')

def get_git_commit_message(cwd=None):
    return subprocess.check_output(['git', 'show', '-s', '--format=%B', 'HEAD'], cwd=cwd).decode('utf8')

def ccap(cmd, dry=False, env=None, **kwargs):
    print_cmd(cmd, dry)
    if not dry:
        subprocess.check_call(cmd, env=env, **kwargs)


MESSAGE_DEPTH = 0

@contextmanager
def timed(msg):
    global MESSAGE_DEPTH #pylint: disable=W0603
    print(colorize('\t'*MESSAGE_DEPTH + '=: ' + msg, color='magenta'))
    tstart = time.time()
    MESSAGE_DEPTH += 1
    yield
    MESSAGE_DEPTH -= 1
    print(colorize('\t'*MESSAGE_DEPTH + "done in %.3f seconds"%(time.time() - tstart), color='magenta'))
