__all__ = ['Monitor', 'get_monitor_files', 'load_results']

import time
from glob import glob
import csv
import json
import os

import gym
from gym.core import Wrapper
import pandas
import uuid


class Monitor(Wrapper):
    EXT = "monitor.csv"
    f = None

    def __init__(self, env, filename, allow_early_resets=False, reset_keywords=(), info_keywords=()):
        """
        A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.

        :param env: (Gym environment) The environment
        :param filename: (str) the location to save a log file, can be None for no log
        :param allow_early_resets: (bool) allows the reset of the environment before it is done
        :param reset_keywords: (tuple) extra keywords for the reset call, if extra parameters are needed at reset
        :param info_keywords: (tuple) extra information to log, from the information return of environment.step
        """
        Wrapper.__init__(self, env=env)
        self.tstart = time.time()
        if filename is None:
            self.f = None
            self.logger = None
        else:
            if not filename.endswith(Monitor.EXT):
                if os.path.isdir(filename):
                    filename = os.path.join(filename, Monitor.EXT)
                else:
                    filename = filename + "." + Monitor.EXT
            self.f = open(filename, "wt")
            self.f.write('#%s\n' % json.dumps({"t_start": self.tstart, 'env_id': env.spec and env.spec.id}))
            self.logger = csv.DictWriter(self.f, fieldnames=('r', 'l', 't') + reset_keywords + info_keywords)
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
        self.current_reset_info = {}  # extra info about the current episode, that was passed in during reset()

    def reset(self, **kwargs):
        """
        Calls the Gym environment reset. Can only be called if the environment is over, or if allow_early_resets is True

        :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
        :return: ([int] or [float]) the first observation of the environment
        """
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError("Tried to reset an environment before done. If you want to allow early resets, "
                               "wrap your env with Monitor(env, path, allow_early_resets=True)")
        self.rewards = []
        self.needs_reset = False
        for k in self.reset_keywords:
            v = kwargs.get(k)
            if v is None:
                raise ValueError('Expected you to pass kwarg %s into reset' % k)
            self.current_reset_info[k] = v
        return self.env.reset(**kwargs)

    def step(self, action):
        """
        Step the environment with the given action

        :param action: ([int] or [float]) the action
        :return: ([int] or [float], [float], [bool], dict) observation, reward, done, information
        """
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
        return ob, rew, done, info

    def close(self):
        """
        Closes the environment
        """
        if self.f is not None:
            self.f.close()

    def get_total_steps(self):
        """
        Returns the total number of timesteps

        :return: (int)
        """
        return self.total_steps

    def get_episode_rewards(self):
        """
        Returns the rewards of all the episodes

        :return: ([float])
        """
        return self.episode_rewards

    def get_episode_lengths(self):
        """
        Returns the number of timesteps of all the episodes

        :return: ([int])
        """
        return self.episode_lengths

    def get_episode_times(self):
        """
        Returns the runtime in seconds of all the episodes

        :return: ([float])
        """
        return self.episode_times


class LoadMonitorResultsError(Exception):
    """
    Raised when loading the monitor log fails.
    """
    pass


def get_monitor_files(path):
    """
    get all the monitor files in the given path

    :param path: (str) the logging folder
    :return: ([str]) the log files
    """
    return glob(os.path.join(path, "*" + Monitor.EXT))


def load_results(path):
    """
    Load results from a given file

    :param path: (str) the path to the log file
    :return: (Pandas DataFrame) the logged data
    """
    monitor_files = (
            glob(os.path.join(path, "*monitor.json")) +
            glob(os.path.join(path, "*monitor.csv")))  # get both csv and (old) json files
    if not monitor_files:
        raise LoadMonitorResultsError("no monitor files of the form *%s found in %s" % (Monitor.EXT, path))
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
            elif fname.endswith('json'):  # Deprecated json format
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
    df.headers = headers  # HACK to preserve backwards compatibility
    return df


def test_monitor():
    """
    test the monitor wrapper
    """
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
    assert set(metadata.keys()) == {'env_id', 'gym_version', 't_start'}, "Incorrect keys in monitor metadata"

    last_logline = pandas.read_csv(f, index_col=None)
    assert set(last_logline.keys()) == {'l', 't', 'r'}, "Incorrect keys in monitor logline"
    f.close()
    os.remove(mon_file)
