import matplotlib.pyplot as plt
import os.path as osp
import json
import os
import numpy as np
import pandas
from collections import defaultdict, namedtuple
from baselines.bench import monitor
from baselines.logger import read_json, read_csv

def smooth(y, radius, mode='two_sided', valid_only=False):
    '''
    Smooth signal y, where radius is determines the size of the window

    mode='twosided':
        average over the window [max(index - radius, 0), min(index + radius, len(y)-1)]
    mode='causal':
        average over the window [max(index - radius, 0), index]

    valid_only: put nan in entries where the full-sized window is not available

    '''
    assert mode in ('two_sided', 'causal')
    if len(y) < 2*radius+1:
        return np.ones_like(y) * y.mean()
    elif mode == 'two_sided':
        convkernel = np.ones(2 * radius+1)
        out = np.convolve(y, convkernel,mode='same') / np.convolve(np.ones_like(y), convkernel, mode='same')
        if valid_only:
            out[:radius] = out[-radius:] = np.nan
    elif mode == 'causal':
        convkernel = np.ones(radius)
        out = np.convolve(y, convkernel,mode='full') / np.convolve(np.ones_like(y), convkernel, mode='full')
        out = out[:-radius+1]
        if valid_only:
            out[:radius] = np.nan
    return out

def one_sided_ema(xolds, yolds, low, high, n, decay_steps=1.):
    luoi = 0 # last unused old index
    sumy = 0.
    county = 0.
    xnews = np.linspace(low, high, n)
    decay_period = (high - low) / (n - 1) * decay_steps
    interstepdecay = np.exp(- 1. / decay_steps)
    sumys = np.zeros_like(xnews)
    countys = np.zeros_like(xnews)
    for i in range(n):
        xnew = xnews[i]
        sumy *= interstepdecay
        county *= interstepdecay
        while True:
            xold = xolds[luoi]
            if xold <= xnew:
                decay = np.exp(- (xnew - xold) / decay_period)
                sumy += decay * yolds[luoi]
                county += decay
                luoi += 1
            else:
                break
            if luoi >= len(xolds):
                break
        sumys[i] = sumy
        countys[i] = county
    return sumys, countys

def smooth_uneven(xolds, yolds, low, high, n, decay_steps=1., mode='symmetric'):
    import pyximport; pyximport.install(setup_args={"include_dirs":np.get_include()})
    # from baselines.common import smooth_helpers #pylint: disable=E0611
    xolds = xolds.astype('float64')
    yolds = yolds.astype('float64')
    if mode == 'causal':
        sumys, countys = one_sided_ema(xolds, yolds, low, high, n, decay_steps)
    elif mode == 'symmetric':
        sumys1, countys1 = one_sided_ema(xolds, yolds, low, high, n, decay_steps)
        sumys2, countys2 = one_sided_ema(-xolds[::-1], yolds[::-1], -high, -low, n, decay_steps)
        sumys = sumys1 + sumys2[::-1]
        countys = countys1 + countys2[::-1]
    xs = np.linspace(low, high, n)
    ys = sumys / countys
    ys[countys < 1e-8] = np.nan
    return xs, ys

def test_smooth():
    norig = 100
    nup = 300
    ndown = 30
    xs = np.cumsum(np.random.rand(norig) * 10 / norig)
    yclean = np.sin(xs)
    ys = yclean + .1 * np.random.randn(yclean.size)
    xup, yup = smooth_uneven(xs, ys, xs.min(), xs.max(), nup, decay_steps=nup/ndown)
    xdown, ydown = smooth_uneven(xs, ys, xs.min(), xs.max(), ndown, decay_steps=ndown/ndown)
    xsame, ysame = smooth_uneven(xs, ys, xs.min(), xs.max(), norig, decay_steps=norig/ndown)
    plt.plot(xs, ys, label='orig', marker='x')
    plt.plot(xup, yup, label='up', marker='x')
    plt.plot(xdown, ydown, label='down', marker='x')
    plt.plot(xsame, ysame, label='same', marker='x')
    plt.plot(xs, yclean, label='clean', marker='x')
    plt.legend()
    plt.show()


Result = namedtuple('Result', 'monitor progress dirname metadata')
Result.__new__.__defaults__ = (None,) * len(Result._fields)

def load_results(root_dir_or_dirs, enable_progress=True, enable_monitor=True, verbose=False):
    '''
    load summaries of runs from a list of directories (including subdirectories)
    Arguments:

    enable_progress: bool - if True, will attempt to load data from progress.csv files (data saved by logger). Default: True
    
    enable_monitor: bool - if True, will attepmt to load data from monitor.csv files (data saved by Monitor environment wrapper). Default: True

    verbose: bool - if True, will print out list of directories from which the data is loaded. Default: False
        

    Returns:
    List of Result objects with the following fields: 
         - dirname - path to the directory data was loaded from
         - metadata - run metadata (such as command-line arguments and anything else in metadata.json file
         - monitor - if enable_monitor is True, this field contains pandas dataframe with loaded monitor.csv file (or aggregate of all *.monitor.csv files in the directory)
         - progress - if enable_progress is True, this field contains pandas dataframe with loaded progress.csv file
    '''
    if isinstance(root_dir_or_dirs, str):
        rootdirs = [osp.expanduser(root_dir_or_dirs)]
    else:
        rootdirs = [osp.expanduser(d) for d in root_dir_or_dirs]
    allresults = []
    for rootdir in rootdirs:
        assert osp.exists(rootdir), "%s doesn't exist"%rootdir
        for dirname, dirs, files in os.walk(rootdir):
            if '-proc' in dirname:
                files[:] = []
                continue
            if set(['metadata.json', 'monitor.json', 'monitor.csv', 'progress.json', 'progress.csv']).intersection(files):
                # used to be uncommented, which means do not go deeper than current directory if any of the data files
                # are found
                # dirs[:] = [] 
                result = {'dirname' : dirname}
                if "metadata.json" in files:
                    with open(osp.join(dirname, "metadata.json"), "r") as fh:
                        result['metadata'] = json.load(fh)
                progjson = osp.join(dirname, "progress.json")
                progcsv = osp.join(dirname, "progress.csv")
                if enable_progress:
                    if osp.exists(progjson):
                        result['progress'] = pandas.DataFrame(read_json(progjson))
                    elif osp.exists(progcsv):
                        try:
                            result['progress'] = read_csv(progcsv)
                        except pandas.errors.EmptyDataError:
                            print('skipping progress file in ', dirname, 'empty data')
                    else:
                        if verbose: print('skipping %s: no progress file'%dirname)

                if enable_monitor:
                    try:
                        result['monitor'] = pandas.DataFrame(monitor.load_results(dirname))
                    except monitor.LoadMonitorResultsError:
                        print('skipping %s: no monitor files'%dirname)
                    except Exception as e:
                        print('exception loading monitor file in %s: %s'%(dirname, e))

                if result.get('monitor') is not None or result.get('progress') is not None: 
                    allresults.append(Result(**result))
                    if verbose:
                        print('successfully loaded %s'%dirname)
                
    if verbose: print('loaded %i results'%len(allresults))
    return allresults

COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
        'brown', 'orange', 'teal',  'lightblue', 'lime', 'lavender', 'turquoise',
        'darkgreen', 'tan', 'salmon', 'gold',  'darkred', 'darkblue']


def default_xy_fn(r):
    x = np.cumsum(r.monitor.l)
    y = smooth(r.monitor.r, radius=10)
    return x,y

def default_split_fn(r):
    import re
    # match name between slash and -<digits> at the end of the string 
    # (slash in the beginning or -<digits> in the end or either may be missing)
    match = re.search(r'[^/-]+(?=(-\d+)?\Z)', r.dirname)
    if match:
        return match.group(0)
    
def plot_results(
    allresults, 
    xy_fn=default_xy_fn,
    split_fn=default_split_fn,
    group_fn=default_split_fn,
    average_group=False,
    figsize=None,
    legend_outside=False,
    resample=0
):
    '''
    plot multiple Results object
    '''

    if split_fn is None: split_fn = lambda _ : ''
    if group_fn is None: group_fn = lambda _ : ''
    sk2r = defaultdict(list) # splitkey2results
    for result in allresults:
        splitkey = split_fn(result)
        sk2r[splitkey].append(result)
    assert len(sk2r) > 0
    assert isinstance(resample, int), "0: don't resample. <integer>: that many samples"
    nrows = len(sk2r)
    ncols = 1
    figsize = figsize or (6, 6 * nrows)
    f, axarr = plt.subplots(nrows, ncols, sharex=False, squeeze=False, figsize=figsize)

    groups = list(set(group_fn(result) for result in allresults))

    for (isplit, sk) in enumerate(sorted(sk2r.keys())):
        g2l = {}
        g2c = defaultdict(int)
        sresults = sk2r[sk]
        gresults = defaultdict(list)
        ax = axarr[isplit][0]
        for result in sresults:
            group = group_fn(result)
            g2c[group] += 1
            x, y = xy_fn(result)
            if x is None: x = np.arange(len(y))
            x, y = map(np.asarray, (x, y))
            if average_group:
                gresults[group].append((x,y))
            else:
                if resample:
                    x, y = smooth_uneven(x, y, x[0], x[-1], resample)
                l, = ax.plot(x, y, color=COLORS[groups.index(group) % len(COLORS)])
                g2l[group] = l
        if average_group:
            for group in sorted(groups):
                xys = gresults[group]
                color = COLORS[groups.index(group)]
                origxs = [xy[0] for xy in xys]
                minxlen = min(map(len, origxs))
                def allequal(qs):
                    return all((q==qs[0]).all() for q in qs[1:])
                if resample:
                    low = 0 # usually right thing to do
                    high = min(x[-1] for x in origxs)
                    usex = np.linspace(low, high, resample)
                    ys = []
                    for (x, y) in xys:
                        ys.append(smooth_uneven(x, y, low, high, resample)[1])
                else:
                    assert allequal([x[:minxlen] for x in origxs]),\
                        'If you want to average unevenly sampled data, set resample=<number of samples you want>'
                    usex = origxs[0]
                    ys = [xy[1][:minxlen] for xy in xys]
                ymean = np.mean(ys, axis=0)
                ystderr = np.std(ys, axis=0) / np.sqrt(len(ys))
                l, = axarr[isplit][0].plot(usex, ymean, color=color)
                g2l[group] = l
                ax.fill_between(usex, ymean-ystderr, ymean+ystderr, color=color, alpha=.3)

        # https://matplotlib.org/users/legend_guide.html
        plt.tight_layout()
        if any(g2l.keys()):
            ax.legend(
                g2l.values(),
                ['%s (%i)'%(g, g2c[g]) for g in g2l] if average_group else g2l.keys(),
                loc=2 if legend_outside else None,
                bbox_to_anchor=(1,1) if legend_outside else None)
        ax.set_title(sk)
    return f, axarr

def regression_analysis(df):
    xcols = list(df.columns.copy())
    xcols.remove('score')
    ycols = ['score']
    import statsmodels.api as sm
    mod = sm.OLS(df[ycols], sm.add_constant(df[xcols]), hasconst=False)
    res = mod.fit()
    print(res.summary())


