import os.path as osp
import numpy as np
from baselines.bench.monitor import load_results

import glob
def count_results(dir):
    # API note: it is important that monitor order dosen't change
    monitor_files = glob(osp.join(dir, "*monitor.*")) # get both csv and (old) json files
    if not monitor_files:
        raise LoadMonitorResultsError("no monitor files of the form *%s found in %s" % (Monitor.EXT, dir))
    names = []
    results = []
    for fname in sorted(monitor_files):  # by string not by numeric value
        with open(fname, 'rt') as fh:
            num_lines = sum(1 for line in fh)
        if fname.endswith('csv'):
            # comment + header
            num_results = max(0, num_lines - 2)
        elif fname.endswith('json'): # Deprecated json format
            num_results = max(0, num_lines - 1)
        names.append(fname)
        results.append(num_results)
    return names, results


class EpisodeCounter:
    def __init__(self, nenvs, bench_dir, num_episodes=20):
        self.first_step = True
        self.bench_dir = bench_dir
        self.num_episodes = 20
        # to obtain unbiased results predetermine which we want to keep
        self.stop_target = np.ones(nenvs, dtype=np.int)*(num_episodes//nenvs)
        self.stop_target[:num_episodes%nenvs] += 1
        assert(self.stop_target.sum() == num_episodes)

    def step(self):
        new_names, episodes_completed = count_results(self.bench_dir)
        if self.first_step:
            self.first_step = False
            self.names = new_names
            assert(sum(episodes_completed) == 0)
        assert(self.names == new_names)
        episodes_completed = np.array(episodes_completed)
        if np.all(episodes_completed >= self.stop_target):
            return False
        return True

    def load_results(self):
        import pandas
        res_keep = []
        for name, target in zip(self.names, self.stop_target):
            glob_str = osp.basename(name)
            res = load_results(self.bench_dir, glob_str)
            res_keep.append(res[:target])
        return pandas.concat(res_keep)

    def save_results(self, filename):
        import pandas
        self.load_results().to_csv(filename)
