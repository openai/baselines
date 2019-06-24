from copy import deepcopy


class Hindsight(object):

    def __init__(self, compute_reward, strategy='final'):
        if callable(compute_reward):
            self.compute_reward = compute_reward
        if strategy == 'final':
            self._get_hindsight_trajectories = self._strategy_final
        else:
            raise ValueError('unknown strategy {}')

    def add(self, trajectories):

        # Note: The reward for the last transiton of the trajectory cannot be computed as the obs after
        # last action is not available. We can ignore this last step and consider the step before it
        # as the last - resulting in a trajectory shorter by a step.

        trajectories = deepcopy(trajectories)
        hs_trajectories = []
        for trajectory in trajectories:
            hs_trajectories.extend(self._get_hindsight_trajectories(trajectory))

        return hs_trajectories

    def _strategy_final(self, trajectory):
        trajectory.pop()   # remove last step as the last obs is not useful in computing reward
        # change the desired goal to last obs' achieved_goal
        trajectory.update_obs('desired_goal', trajectory.obs[-1]['achieved_goal'].copy())
        trajectory.update_rewards(self.compute_reward)
        return [trajectory]

    def _strategy_future(self, trajectory):

        raise NotImplementedError

