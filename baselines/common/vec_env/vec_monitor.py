from . import VecEnvWrapper
from baselines.bench.monitor import ResultsWriter
import numpy as np
import time


class VecMonitor(VecEnvWrapper):
    def __init__(self, venv, filename=None):
        VecEnvWrapper.__init__(self, venv)
        self.eprets = None
        self.eplens = None
        self.tstart = time.time()
        self.results_writer = ResultsWriter(filename, header={'t_start': self.tstart})

    def reset(self):
        obs = self.venv.reset()
        self.eprets = np.zeros(self.num_envs, 'f')
        self.eplens = np.zeros(self.num_envs, 'i')
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self.eprets += rews
        self.eplens += 1
        newinfos = []
        for (i, (done, ret, eplen, info)) in enumerate(zip(dones, self.eprets, self.eplens, infos)):
            info = info.copy()
            if done:
                epinfo = {'r': ret, 'l': eplen, 't': round(time.time() - self.tstart, 6)}
                info['episode'] = epinfo
                self.eprets[i] = 0
                self.eplens[i] = 0
                self.results_writer.write_row(epinfo)

            newinfos.append(info)

        return obs, rews, dones, newinfos
