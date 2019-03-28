__all__ = ['Monitor', 'get_monitor_files', 'load_results']

import csv
import json
import os
import time
from glob import glob

import pandas
from gym.core import Wrapper


class Monitor(Wrapper):
    EXT = "monitor.csv"
    file_handler = None

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
        self.t_start = time.time()
        if filename is None:
            self.file_handler = None
            self.logger = None
        else:
            if not filename.endswith(Monitor.EXT):
                if os.path.isdir(filename):
                    filename = os.path.join(filename, Monitor.EXT)
                else:
                    filename = filename + "." + Monitor.EXT
            self.file_handler = open(filename, "wt")
            self.file_handler.write('#%s\n' % json.dumps({"t_start": self.t_start, 'env_id': env.spec and env.spec.id}))
            self.logger = csv.DictWriter(self.file_handler,
                                         fieldnames=('r', 'l', 't') + reset_keywords + info_keywords)
            self.logger.writeheader()
            self.file_handler.flush()

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
        for key in self.reset_keywords:
            value = kwargs.get(key)
            if value is None:
                raise ValueError('Expected you to pass kwarg %s into reset' % key)
            self.current_reset_info[key] = value
        return self.env.reset(**kwargs)

    def step(self, action):
        """
        Step the environment with the given action

        :param action: ([int] or [float]) the action
        :return: ([int] or [float], [float], [bool], dict) observation, reward, done, information
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, done, info = self.env.step(action)
        self.rewards.append(reward)
        if done:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            eplen = len(self.rewards)
            ep_info = {"r": round(ep_rew, 6), "l": eplen, "t": round(time.time() - self.t_start, 6)}
            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_rewards.append(ep_rew)
            self.episode_lengths.append(eplen)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.logger:
                self.logger.writerow(ep_info)
                self.file_handler.flush()
            info['episode'] = ep_info
        self.total_steps += 1
        return observation, reward, done, info

    def close(self):
        """
        Closes the environment
        """
        if self.file_handler is not None:
            self.file_handler.close()

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
    # get both csv and (old) json files
    monitor_files = (glob(os.path.join(path, "*monitor.json")) + glob(os.path.join(path, "*monitor.csv")))
    if not monitor_files:
        raise LoadMonitorResultsError("no monitor files of the form *%s found in %s" % (Monitor.EXT, path))
    data_frames = []
    headers = []
    for file_name in monitor_files:
        with open(file_name, 'rt') as file_handler:
            if file_name.endswith('csv'):
                first_line = file_handler.readline()
                assert first_line[0] == '#'
                header = json.loads(first_line[1:])
                data_frame = pandas.read_csv(file_handler, index_col=None)
                headers.append(header)
            elif file_name.endswith('json'):  # Deprecated json format
                episodes = []
                lines = file_handler.readlines()
                header = json.loads(lines[0])
                headers.append(header)
                for line in lines[1:]:
                    episode = json.loads(line)
                    episodes.append(episode)
                data_frame = pandas.DataFrame(episodes)
            else:
                assert 0, 'unreachable'
            data_frame['t'] += header['t_start']
        data_frames.append(data_frame)
    data_frame = pandas.concat(data_frames)
    data_frame.sort_values('t', inplace=True)
    data_frame.reset_index(inplace=True)
    data_frame['t'] -= min(header['t_start'] for header in headers)
    # data_frame.headers = headers  # HACK to preserve backwards compatibility
    return data_frame
