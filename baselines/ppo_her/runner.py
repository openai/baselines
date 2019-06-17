import numpy as np
from baselines.common.runners import AbstractEnvRunner
from collections import namedtuple

class Trajectory(object):
    def __init__(self):
        self.obs = []
        self.actions = []
        self.dones = []

    def append(self, obs, action, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.dones.append(done)

class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, nsteps):
        super().__init__(env=env, model=model, nsteps=nsteps)

    def run(self):
        # Here, we init the lists that will contain the trajectories
        trajectory = Trajectory()
        epinfos = []
        # For n in range number of steps
        for _ in range(self.nsteps):
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, _ , self.states, _ = self.model.step(self.obs, S=self.states, M=self.dones)

            # self.model.step - retuns

            trajectory.obs.append(self.c)
            trajs['obs'].append(self.obs.copy())
            trajs['actions'].append(actions)
            trajs['dones'].append(self.dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)

        #batch of steps to batch of rollouts
        #TODO: converting to np.array might not be needed
        trajs['obs'] = np.asarray(trajs['obs'], dtype=self.obs.dtype)
        trajs['actions'] = np.asarray(trajs['actions'])
        trajs['dones'] = np.asarray(trajs['dones'], dtype=np.bool)
        trajs['epinfos'] = epinfos

        return trajs


