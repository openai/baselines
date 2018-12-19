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

def one_sided_ema(xolds, yolds, low=None, high=None, n=512, decay_steps=1., low_counts_threshold=1e-8):
    '''
    perform one-sided (causal) EMA (exponential moving average)
    smoothing and resampling to an even grid with n points.
    Does not do extrapolation, so we assume
    xolds[0] <= low && high <= xolds[-1]

    Arguments:

    xolds: array or list  - x values of data. Needs to be sorted in ascending order
    yolds: array of list  - y values of data. Has to have the same length as xolds

    low: float            - min value of the new x grid. By default equals to xolds[0]
    high: float           - max value of the new x grid. By default equals to xolds[-1]

    n: int                - number of points in new x grid

    decay_steps: float    - EMA decay factor, expressed in new x grid steps.

    low_counts_threshold: float or int
                          - y values with counts less than this value will be set to NaN

    Returns:
        tuple sum_ys, count_ys where
            xs        - array with new x grid
            ys        - array of EMA of y at each point of the new x grid
            count_ys  - array of EMA of y counts at each point of the new x grid

    '''

    low = xolds[0] if low is None else low
    high = xolds[-1] if high is None else high

    assert xolds[0] <= low, 'low = {} < xolds[0] = {} - extrapolation not permitted!'.format(low, xolds[0])
    assert xolds[-1] >= high, 'high = {} > xolds[-1] = {}  - extrapolation not permitted!'.format(high, xolds[-1])
    assert len(xolds) == len(yolds), 'length of xolds ({}) and yolds ({}) do not match!'.format(len(xolds), len(yolds))


    xolds = xolds.astype('float64')
    yolds = yolds.astype('float64')

    luoi = 0 # last unused old index
    sum_y = 0.
    count_y = 0.
    xnews = np.linspace(low, high, n)
    decay_period = (high - low) / (n - 1) * decay_steps
    interstep_decay = np.exp(- 1. / decay_steps)
    sum_ys = np.zeros_like(xnews)
    count_ys = np.zeros_like(xnews)
    for i in range(n):
        xnew = xnews[i]
        sum_y *= interstep_decay
        count_y *= interstep_decay
        while True:
            xold = xolds[luoi]
            if xold <= xnew:
                decay = np.exp(- (xnew - xold) / decay_period)
                sum_y += decay * yolds[luoi]
                count_y += decay
                luoi += 1
            else:
                break
            if luoi >= len(xolds):
                break
        sum_ys[i] = sum_y
        count_ys[i] = count_y

    ys = sum_ys / count_ys
    ys[count_ys < low_counts_threshold] = np.nan

    return xnews, ys, count_ys

def symmetric_ema(xolds, yolds, low=None, high=None, n=512, decay_steps=1., low_counts_threshold=1e-8):
    '''
    perform symmetric EMA (exponential moving average)
    smoothing and resampling to an even grid with n points.
    Does not do extrapolation, so we assume
    xolds[0] <= low && high <= xolds[-1]

    Arguments:

    xolds: array or list  - x values of data. Needs to be sorted in ascending order
    yolds: array of list  - y values of data. Has to have the same length as xolds

    low: float            - min value of the new x grid. By default equals to xolds[0]
    high: float           - max value of the new x grid. By default equals to xolds[-1]

    n: int                - number of points in new x grid

    decay_steps: float    - EMA decay factor, expressed in new x grid steps.

    low_counts_threshold: float or int
                          - y values with counts less than this value will be set to NaN

    Returns:
        tuple sum_ys, count_ys where
            xs        - array with new x grid
            ys        - array of EMA of y at each point of the new x grid
            count_ys  - array of EMA of y counts at each point of the new x grid

    '''
    xs, ys1, count_ys1 = one_sided_ema(xolds, yolds, low, high, n, decay_steps, low_counts_threshold=0)
    _,  ys2, count_ys2 = one_sided_ema(-xolds[::-1], yolds[::-1], -high, -low, n, decay_steps, low_counts_threshold=0)
    ys2 = ys2[::-1]
    count_ys2 = count_ys2[::-1]
    count_ys = count_ys1 + count_ys2
    ys = (ys1 * count_ys1 + ys2 * count_ys2) / count_ys
    ys[count_ys < low_counts_threshold] = np.nan
    return xs, ys, count_ys

Result = namedtuple('Result', 'monitor progress dirname metadata')
Result.__new__.__defaults__ = (None,) * len(Result._fields)

def load_results(root_dir_or_dirs, enable_progress=True, enable_monitor=True, verbose=False):
    '''
    load summaries of runs from a list of directories (including subdirectories)
    Arguments:

    enable_progress: bool - if True, will attempt to load data from progress.csv files (data saved by logger). Default: True

    enable_monitor: bool - if True, will attempt to load data from monitor.csv files (data saved by Monitor environment wrapper). Default: True

    verbose: bool - if True, will print out list of directories from which the data is loaded. Default: False


    Returns:
    List of Result objects with the following fields:
         - dirname - path to the directory data was loaded from
         - metadata - run metadata (such as command-line arguments and anything else in metadata.json file
         - monitor - if enable_monitor is True, this field contains pandas dataframe with loaded monitor.csv file (or aggregate of all *.monitor.csv files in the directory)
         - progress - if enable_progress is True, this field contains pandas dataframe with loaded progress.csv file
    '''
    import re
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
            monitor_re = re.compile(r'(\d+\.)?(\d+\.)?monitor\.csv')
            if set(['metadata.json', 'monitor.json', 'progress.json', 'progress.csv']).intersection(files) or \
               any([f for f in files if monitor_re.match(f)]):  # also match monitor files like 0.1.monitor.csv
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
    allresults, *,
    xy_fn=default_xy_fn,
    split_fn=default_split_fn,
    group_fn=default_split_fn,
    average_group=False,
    shaded_std=True,
    shaded_err=True,
    figsize=None,
    legend_outside=False,
    resample=0,
    smooth_step=1.0,
):
    '''
    Plot multiple Results objects

    xy_fn: function Result -> x,y           - function that converts results objects into tuple of x and y values.
                                              By default, x is cumsum of episode lengths, and y is episode rewards

    split_fn: function Result -> hashable   - function that converts results objects into keys to split curves into sub-panels by.
                                              That is, the results r for which split_fn(r) is different will be put on different sub-panels.
                                              By default, the portion of r.dirname between last / and -<digits> is returned. The sub-panels are
                                              stacked vertically in the figure.

    group_fn: function Result -> hashable   - function that converts results objects into keys to group curves by.
                                              That is, the results r for which group_fn(r) is the same will be put into the same group.
                                              Curves in the same group have the same color (if average_group is False), or averaged over
                                              (if average_group is True). The default value is the same as default value for split_fn

    average_group: bool                     - if True, will average the curves in the same group and plot the mean. Enables resampling
                                              (if resample = 0, will use 512 steps)

    shaded_std: bool                        - if True (default), the shaded region corresponding to standard deviation of the group of curves will be
                                              shown (only applicable if average_group = True)

    shaded_err: bool                        - if True (default), the shaded region corresponding to error in mean estimate of the group of curves
                                              (that is, standard deviation divided by square root of number of curves) will be
                                              shown (only applicable if average_group = True)

    figsize: tuple or None                  - size of the resulting figure (including sub-panels). By default, width is 6 and height is 6 times number of
                                              sub-panels.


    legend_outside: bool                    - if True, will place the legend outside of the sub-panels.

    resample: int                           - if not zero, size of the uniform grid in x direction to resample onto. Resampling is performed via symmetric
                                              EMA smoothing (see the docstring for symmetric_ema).
                                              Default is zero (no resampling). Note that if average_group is True, resampling is necessary; in that case, default
                                              value is 512.

    smooth_step: float                      - when resampling (i.e. when resample > 0 or average_group is True), use this EMA decay parameter (in units of the new grid step).
                                              See docstrings for decay_steps in symmetric_ema or one_sided_ema functions.

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

    default_samples = 512
    if average_group:
        resample = resample or default_samples

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
                    x, y, counts = symmetric_ema(x, y, x[0], x[-1], resample, decay_steps=smooth_step)
                l, = ax.plot(x, y, color=COLORS[groups.index(group) % len(COLORS)])
                g2l[group] = l
        if average_group:
            for group in sorted(groups):
                xys = gresults[group]
                if not any(xys):
                    continue
                color = COLORS[groups.index(group) % len(COLORS)]
                origxs = [xy[0] for xy in xys]
                minxlen = min(map(len, origxs))
                def allequal(qs):
                    return all((q==qs[0]).all() for q in qs[1:])
                if resample:
                    low  = max(x[0] for x in origxs)
                    high = min(x[-1] for x in origxs)
                    usex = np.linspace(low, high, resample)
                    ys = []
                    for (x, y) in xys:
                        ys.append(symmetric_ema(x, y, low, high, resample, decay_steps=smooth_step)[1])
                else:
                    assert allequal([x[:minxlen] for x in origxs]),\
                        'If you want to average unevenly sampled data, set resample=<number of samples you want>'
                    usex = origxs[0]
                    ys = [xy[1][:minxlen] for xy in xys]
                ymean = np.mean(ys, axis=0)
                ystd = np.std(ys, axis=0)
                ystderr = ystd / np.sqrt(len(ys))
                l, = axarr[isplit][0].plot(usex, ymean, color=color)
                g2l[group] = l
                if shaded_err:
                    ax.fill_between(usex, ymean - ystderr, ymean + ystderr, color=color, alpha=.4)
                if shaded_std:
                    ax.fill_between(usex, ymean - ystd,    ymean + ystd,    color=color, alpha=.2)


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

def test_smooth():
    norig = 100
    nup = 300
    ndown = 30
    xs = np.cumsum(np.random.rand(norig) * 10 / norig)
    yclean = np.sin(xs)
    ys = yclean + .1 * np.random.randn(yclean.size)
    xup, yup, _ = symmetric_ema(xs, ys, xs.min(), xs.max(), nup, decay_steps=nup/ndown)
    xdown, ydown, _ = symmetric_ema(xs, ys, xs.min(), xs.max(), ndown, decay_steps=ndown/ndown)
    xsame, ysame, _ = symmetric_ema(xs, ys, xs.min(), xs.max(), norig, decay_steps=norig/ndown)
    plt.plot(xs, ys, label='orig', marker='x')
    plt.plot(xup, yup, label='up', marker='x')
    plt.plot(xdown, ydown, label='down', marker='x')
    plt.plot(xsame, ysame, label='same', marker='x')
    plt.plot(xs, yclean, label='clean', marker='x')
    plt.legend()
    plt.show()


