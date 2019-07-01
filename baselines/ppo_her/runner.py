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
        trajectories_rewards = []
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
            trajectories_rewards.append(rewards)
        trajectories_obs.append(self.obs.copy())
        trajectories_dones.append(self.dones)

        # Here, we collect trajectories by appropriately splitting the sequence for each env
        # In a trajectory num_obs = num_actions + 1 (for non-terminating episode the final obs is required in
        # calcualting last value for GAE)
        trajectories = []
        for env_idx in range(self.nenv):
            trajectory = Trajectory()
            for t in range(self.nsteps):
                obs, action, nextobs, rewards, nextterminal =\
                    step(trajectories_obs, trajectories_actions, trajectories_rewards, trajectories_dones,
                         env_idx, t)
                trajectory.obs.append(obs)
                trajectory.actions.append(action)
                trajectory.rewards.append(rewards)
                if nextterminal:
                    trajectory.obs.append(nextobs)
                    trajectories.append(trajectory)
                    trajectory.done = True
                    trajectory = Trajectory()
            if not nextterminal:    # add last trajectory if it is non-terminal
                trajectory.obs.append(nextobs)
                trajectory.done = False
                trajectories.append(trajectory)
        return trajectories, epinfos


def step(trajectories_obs, trajectories_actions, trajectories_rewards, trajectories_dones, env_idx, t):
    """ helper function to return the transition tuple """

    def get_obs(t):
        if t not in range(len(trajectories_obs)):
            return None
        obs = {}
        for key in trajectories_obs[t].keys():
            obs[key] = trajectories_obs[t][key][env_idx:env_idx+1, :]
        return obs

    def get_action(t):
        return trajectories_actions[t][env_idx:env_idx+1, :]

    def get_reward(t):
        return trajectories_rewards[t][env_idx]

    def get_done(t):
        return trajectories_dones[t][env_idx]

    return get_obs(t), get_action(t), get_obs(t + 1), get_reward(t), get_done(t + 1)
