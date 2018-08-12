__all__ = ['Monitor', 'get_monitor_files', 'load_results', 'SimpleMonitor']

import gym
from gym.core import Wrapper
import time
from glob import glob
import csv
import os.path as osp
import json
import numpy as np

class Monitor(Wrapper):
    EXT = "monitor.csv"
    f = None

    def __init__(self, env, filename, allow_early_resets=False, reset_keywords=(), info_keywords=()):
        Wrapper.__init__(self, env=env)
        self.tstart = time.time()
        if filename is None:
            self.f = None
            self.logger = None
        else:
            if not filename.endswith(Monitor.EXT):
                if osp.isdir(filename):
                    filename = osp.join(filename, Monitor.EXT)
                else:
                    filename = filename + "." + Monitor.EXT
            self.f = open(filename, "wt")
            self.f.write('#%s\n'%json.dumps({"t_start": self.tstart, 'env_id' : env.spec and env.spec.id}))
            self.logger = csv.DictWriter(self.f, fieldnames=('r', 'l', 't')+reset_keywords+info_keywords)
            self.logger.writeheader()
            self.f.flush()

        self.reset_keywords = reset_keywords
        self.info_keywords = info_keywords
        self.allow_early_resets = allow_early_resets
        self.rewards = None
        self.needs_reset = True
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0
        self.current_reset_info = {} # extra info about the current episode, that was passed in during reset()

    def reset(self, **kwargs):
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError("Tried to reset an environment before done. If you want to allow early resets, wrap your env with Monitor(env, path, allow_early_resets=True)")
        self.rewards = []
        self.needs_reset = False
        for k in self.reset_keywords:
            v = kwargs.get(k)
            if v is None:
                raise ValueError('Expected you to pass kwarg %s into reset'%k)
            self.current_reset_info[k] = v
        return self.env.reset(**kwargs)

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        ob, rew, done, info = self.env.step(action)
        self.rewards.append(rew)
        if done:
            self.needs_reset = True
            eprew = sum(self.rewards)
            eplen = len(self.rewards)
            epinfo = {"r": round(eprew, 6), "l": eplen, "t": round(time.time() - self.tstart, 6)}
            for k in self.info_keywords:
                epinfo[k] = info[k]
            self.episode_rewards.append(eprew)
            self.episode_lengths.append(eplen)
            self.episode_times.append(time.time() - self.tstart)
            epinfo.update(self.current_reset_info)
            if self.logger:
                self.logger.writerow(epinfo)
                self.f.flush()
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

    def get_episode_times(self):
        return self.episode_times

class LoadMonitorResultsError(Exception):
    pass

def get_monitor_files(dir):
    return glob(osp.join(dir, "*" + Monitor.EXT))

def load_results(dir):
    import pandas
    monitor_files = (
        glob(osp.join(dir, "*monitor.json")) + 
        glob(osp.join(dir, "*monitor.csv"))) # get both csv and (old) json files
    if not monitor_files:
        raise LoadMonitorResultsError("no monitor files of the form *%s found in %s" % (Monitor.EXT, dir))
    dfs = []
    headers = []
    for fname in monitor_files:
        with open(fname, 'rt') as fh:
            if fname.endswith('csv'):
                firstline = fh.readline()
                assert firstline[0] == '#'
                header = json.loads(firstline[1:])
                df = pandas.read_csv(fh, index_col=None)
                headers.append(header)
            elif fname.endswith('json'): # Deprecated json format
                episodes = []
                lines = fh.readlines()
                header = json.loads(lines[0])
                headers.append(header)
                for line in lines[1:]:
                    episode = json.loads(line)
                    episodes.append(episode)
                df = pandas.DataFrame(episodes)
            else:
                assert 0, 'unreachable'
            df['t'] += header['t_start']
        dfs.append(df)
    df = pandas.concat(dfs)
    df.sort_values('t', inplace=True)
    df.reset_index(inplace=True)
    df['t'] -= min(header['t_start'] for header in headers)
    df.headers = headers # HACK to preserve backwards compatibility
    return df


 class SimpleMonitor(Wrapper):
    def __init__(self, env):
        """Adds two qunatities to info returned by every step:

            num_steps: int
                Number of steps takes so far
            rewards: [float]
                All the cumulative rewards for the episodes completed so far.
        """
        super().__init__(env)
        # current episode state
        self._current_reward = None
        self._num_steps = None
        # temporary monitor state that we do not save
        self._time_offset = None
        self._total_steps = None
        # monitor state
        self._episode_rewards = []
        self._episode_lengths = []
        self._episode_end_times = []

    def _reset(self):
        obs = self.env.reset()
        # recompute temporary state if needed
        if self._time_offset is None:
            self._time_offset = time.time()
            if len(self._episode_end_times) > 0:
                self._time_offset -= self._episode_end_times[-1]
        if self._total_steps is None:
            self._total_steps = sum(self._episode_lengths)
        # update monitor state
        if self._current_reward is not None:
            self._episode_rewards.append(self._current_reward)
            self._episode_lengths.append(self._num_steps)
            self._episode_end_times.append(time.time() - self._time_offset)
        # reset episode state
        self._current_reward = 0
        self._num_steps = 0

        return obs

    def _step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._current_reward += rew
        self._num_steps += 1
        self._total_steps += 1
        info['steps'] = self._total_steps
        info['rewards'] = self._episode_rewards
        return (obs, rew, done, info)

    def get_state(self):
        return {
            'env_id': self.env.unwrapped.spec.id,
            'episode_data': {
                'episode_rewards': self._episode_rewards,
                'episode_lengths': self._episode_lengths,
                'episode_end_times': self._episode_end_times,
                'initial_reset_time': 0,
            }
        }

    def set_state(self, state):
        assert state['env_id'] == self.env.unwrapped.spec.id
        ed = state['episode_data']
        self._episode_rewards = ed['episode_rewards']
        self._episode_lengths = ed['episode_lengths']
        self._episode_end_times = ed['episode_end_times']

        
def test_monitor():
    env = gym.make("CartPole-v1")
    env.seed(0)
    mon_file = "/tmp/baselines-test-%s.monitor.csv" % uuid.uuid4()
    menv = Monitor(env, mon_file)
    menv.reset()
    for _ in range(1000):
        _, _, done, _ = menv.step(0)
        if done:
            menv.reset()

    f = open(mon_file, 'rt')

    firstline = f.readline()
    assert firstline.startswith('#')
    metadata = json.loads(firstline[1:])
    assert metadata['env_id'] == "CartPole-v1"
    assert set(metadata.keys()) == {'env_id', 'gym_version', 't_start'},  "Incorrect keys in monitor metadata"

    last_logline = pandas.read_csv(f, index_col=None)
    assert set(last_logline.keys()) == {'l', 't', 'r'}, "Incorrect keys in monitor logline"
    f.close()
    os.remove(mon_file)