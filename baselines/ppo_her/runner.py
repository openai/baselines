import numpy as np
from gym.core import GoalEnv

class Runner(object):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, nsteps):
        self.env = env  # env must be a GoalEnv
        self.model = model
        self.nenv = nenv = env.num_envs if hasattr(env, 'num_envs') else 1
        self.obs = env.reset()
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    def obsasarray(self, obs_dict):
        obs = []
        for key in self.env.observation_space.spaces.keys():
            obs.append(obs_dict[key])
        return np.concatenate(obs, axis=1)

    def run(self):
        # Here, we init the lists that will contain the trajectories
        trajs_obs = []
        trajs_actions = []
        trajs_dones = []
        epinfos = []
        # For n in range number of steps
        for n in range(self.nsteps):
            # We already have self.obs because Runner superclass run self.obs = env.reset() on init
            # For reference: actions, values, states, neglogpacs = model.step()
            actions, _, self.states, _ = self.model.step(self.obsasarray(self.obs), S=self.states, M=self.dones)
            trajs_obs.append(self.obs.copy())
            trajs_actions.append(actions)
            trajs_dones.append(self.dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs, rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
        trajs_obs.append(self.obs.copy())   # need the final obs to compute reward

        return trajs_obs, trajs_actions, trajs_dones



