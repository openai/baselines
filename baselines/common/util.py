import subprocess
import shlex

def colorize(string, color='green', bold=False, highlight=False):
    """
    Return string surrounded by appropriate terminal color codes to
    print colorized text.  Valid colors: gray, red, green, yellow,
    blue, magenta, cyan, white, crimson
    """
    attr = []
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
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append('1')
    attrs = ';'.join(attr)
    return '\x1b[%sm%s\x1b[0m' % (attrs, string)


def print_cmd(cmd, dry=False):
    if isinstance(cmd, str):  # for shell=True
        pass
    else:
        cmd = ' '.join(shlex.quote(arg) for arg in cmd)
    print(colorize(('CMD: ' if not dry else 'DRY: ') + cmd))


# this is here because a few other packages outside of rcall depend on it, but it should be removed
def get_git_commit(cwd=None):
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=cwd).decode('utf8')

def ccap(cmd, dry=False, env=None, **kwargs):
    print_cmd(cmd, dry)
    if not dry:
        subprocess.check_call(cmd, env=env, **kwargs)


