__all__ = ['Monitor', 'get_monitor_files', 'load_results']

import gym
from gym.core import Wrapper
from os import path
import time
from glob import glob

try:
    import ujson as json # Not necessary for monitor writing, but very useful for monitor loading
except ImportError:
    import json

class Monitor(Wrapper):
    EXT = "monitor.json"
    f = None

    def __init__(self, env, filename, allow_early_resets=False):
        Wrapper.__init__(self, env=env)
        self.tstart = time.time()
        if filename is None:
            self.f = None
            self.logger = None
        else:
            if not filename.endswith(Monitor.EXT):
                filename = filename + "." + Monitor.EXT
            self.f = open(filename, "wt")
            self.logger = JSONLogger(self.f)
            self.logger.writekvs({"t_start": self.tstart, "gym_version": gym.__version__,
                "env_id": env.spec.id if env.spec else 'Unknown'})
        self.allow_early_resets = allow_early_resets
        self.rewards = None
        self.needs_reset = True
        self.episode_rewards = []
        self.episode_lengths = []
        self.total_steps = 0
        self.current_metadata = {} # extra info that gets injected into each log entry
        # Useful for metalearning where we're modifying the environment externally
        # But want our logs to know about these modifications

    def __getstate__(self): # XXX
        d = self.__dict__.copy()
        if self.f:
            del d['f'], d['logger']
            d['_filename'] = self.f.name
            d['_num_episodes'] = len(self.episode_rewards)
        else:
            d['_filename'] = None
        return d
    def __setstate__(self, d):
        filename = d.pop('_filename')
        self.__dict__ = d
        if filename is not None:
            nlines = d.pop('_num_episodes') + 1
            self.f = open(filename, "r+t")
            for _ in range(nlines):
                self.f.readline()
            self.f.truncate()        
            self.logger = JSONLogger(self.f)


    def reset(self):
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError("Tried to reset an environment before done. If you want to allow early resets, wrap your env with Monitor(env, path, allow_early_resets=True)")
        self.rewards = []
        self.needs_reset = False
        return self.env.reset()

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        ob, rew, done, info = self.env.step(action)
        self.rewards.append(rew)
        if done:
            self.needs_reset = True
            eprew = sum(self.rewards)
            eplen = len(self.rewards)
            epinfo = {"r": eprew, "l": eplen, "t": round(time.time() - self.tstart, 6)}
            epinfo.update(self.current_metadata)
            if self.logger:
                self.logger.writekvs(epinfo)
            self.episode_rewards.append(eprew)
            self.episode_lengths.append(eplen)
            info['episode'] = epinfo
        self.total_steps += 1
        return (ob, rew, done, info)

    def close(self):
        if self.f is not None:
            self.f.close()

    def get_total_steps(self):
        return self.total_steps

    def get_episode_rewards(self):
        return self.episode_rewards

    def get_episode_lengths(self):
        return self.episode_lengths

class JSONLogger(object):
    def __init__(self, file):
        self.file = file

    def writekvs(self, kvs):
        for k,v in kvs.items():
            if hasattr(v, 'dtype'):
                v = v.tolist()
                kvs[k] = float(v)
        self.file.write(json.dumps(kvs) + '\n')
        self.file.flush()


class LoadMonitorResultsError(Exception):
    pass

def get_monitor_files(dir):
    return glob(path.join(dir, "*" + Monitor.EXT))

def load_results(dir):
    fnames = get_monitor_files(dir)
    if not fnames:
        raise LoadMonitorResultsError("no monitor files of the form *%s found in %s" % (Monitor.EXT, dir))
    episodes = []
    headers = []
    for fname in fnames:
        with open(fname, 'rt') as fh:
            lines = fh.readlines()
        header = json.loads(lines[0])
        headers.append(header)
        for line in lines[1:]:
            episode = json.loads(line)
            episode['abstime'] = header['t_start'] + episode['t']
            del episode['t']
            episodes.append(episode)
    header0 = headers[0]
    for header in headers[1:]:
        assert header['env_id'] == header0['env_id'], "mixing data from two envs"
    episodes = sorted(episodes, key=lambda e: e['abstime'])
    return {
        'env_info': {'env_id': header0['env_id'], 'gym_version': header0['gym_version']},
        'episode_end_times': [e['abstime'] for e in episodes],
        'episode_lengths': [e['l'] for e in episodes],
        'episode_rewards': [e['r'] for e in episodes],
        'initial_reset_time': min([min(header['t_start'] for header in headers)])
    }
