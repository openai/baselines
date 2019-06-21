import numpy as np
from gym.core import GoalEnv
from baselines.ppo_her.traj_util import Trajectory


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

    def run(self):
        """ run the env using the current model and return collected trajectories """
        # Here, we init the lists that will contain the trajectories
        trajectories_obs = []
        trajectories_actions = []
        trajectories_dones = []
        epinfos = []
        # For n in range number of steps
        obsasarray = self.env.observation_space.to_array  # maintains key order while concatenation
        for _ in range(self.nsteps):
            # We already have self.obs because Runner superclass run self.obs = env.reset() on init
            # For reference: actions, values, states, neglogpacs = model.step()
            actions, _, self.states, _ = self.model.step(obsasarray(self.obs), S=self.states, M=self.dones)
            trajectories_obs.append(self.obs.copy())
            trajectories_actions.append(actions)
            trajectories_dones.append(self.dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs, rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
        trajectories_obs.append(self.obs.copy())   # need the final obs to compute reward
        trajectories_dones.append(self.dones)      # needed compute last value

        # Here, we collect trajectories by appropriately splitting the sequence for each env
        trajectories = []
        for env_index in range(self.nenv):
            new_trajectory = True
            for step in range(self.nsteps):
                obs, action, nextobs, nextterminal =\
                    transition(trajectories_obs, trajectories_actions, trajectories_dones, env_index, step)
                if new_trajectory:
                    trajectory = Trajectory()
                    new_trajectory = False
                trajectory.obs.append(obs)
                trajectory.actions.append(action)
                if nextterminal:
                    trajectory.obs.append(nextobs)
                    new_trajectory = True
                    trajectories.append(trajectory)
        return trajectories, epinfos


def transition(trajectories_obs, trajectories_actions, trajectories_dones, env_idx, i):
    """ helper function to return the transition tuple """

    def get_obs(i):
        obs = {}
        for key in trajectories_obs[i].keys():
            obs[key] = trajectories_obs[i][key][env_idx]
        return obs

    def get_action(i):
        return trajectories_actions[i][env_idx]

    def get_done(i):
        return trajectories_dones[i][env_idx]

    return get_obs(i), get_action(i), get_obs(i + 1), get_done(i + 1)
