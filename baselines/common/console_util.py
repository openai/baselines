from __future__ import print_function
from contextlib import contextmanager
import numpy as np
import time

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
    if isinstance(x, float): rep = "%g"%x
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

def colorize(string, color, bold=False, highlight=False):
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


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
