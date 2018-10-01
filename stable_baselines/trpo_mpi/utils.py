import gym
import numpy as np

from stable_baselines.common.vec_env import VecEnv


def traj_segment_generator(policy, env, horizon, reward_giver=None, gail=False):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)

    :param policy: (MLPPolicy) the policy
    :param env: (Gym Environment) the environment
    :param horizon: (int) the number of timesteps to run per batch
    :param reward_giver: (TransitionClassifier) the reward predicter from obsevation and action
    :param gail: (bool) Whether we are using this generator for standard trpo or with gail
    :return: (dict) generator that returns a dict with the following keys:

        - ob: (np.ndarray) observations
        - rew: (numpy float) rewards (if gail is used it is the predicted reward)
        - vpred: (numpy float) action logits
        - new: (numpy bool) dones (is end of episode)
        - ac: (np.ndarray) actions
        - prevac: (np.ndarray) previous actions
        - nextvpred: (numpy float) next action logits
        - ep_rets: (float) cumulated current episode reward
        - ep_lens: (int) the length of the current episode
        - ep_true_rets: (float) the real environment reward
    """
    # Check when using GAIL
    assert not (gail and reward_giver is None), "You must pass a reward giver when using GAIL"

    # Initialize state variables
    step = 0
    action = env.action_space.sample()  # not used, just so we have the datatype
    new = True
    observation = env.reset()

    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode
    cur_ep_true_ret = 0
    ep_true_rets = []
    ep_rets = []  # returns of completed episodes in this segment
    ep_lens = []  # Episode lengths

    # Initialize history arrays
    observations = np.array([observation for _ in range(horizon)])
    true_rews = np.zeros(horizon, 'float32')
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    dones = np.zeros(horizon, 'int32')
    actions = np.array([action for _ in range(horizon)])
    prev_actions = actions.copy()
    states = policy.initial_state
    done = None

    while True:
        prevac = action
        action, vpred, states, _ = policy.step(observation.reshape(-1, *observation.shape), states, done)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if step > 0 and step % horizon == 0:
            # Fix to avoid "mean of empty slice" warning when there is only one episode
            if len(ep_rets) == 0:
                ep_rets = [cur_ep_ret]
                ep_lens = [cur_ep_len]
                ep_true_rets = [cur_ep_true_ret]
                total_timesteps = cur_ep_len
            else:
                total_timesteps = sum(ep_lens) + cur_ep_len

            yield {"ob": observations, "rew": rews, "dones": dones, "true_rew": true_rews, "vpred": vpreds,
                   "ac": actions, "prevac": prev_actions, "nextvpred": vpred * (1 - new), "ep_rets": ep_rets,
                   "ep_lens": ep_lens, "ep_true_rets": ep_true_rets, "total_timestep": total_timesteps}
            _, vpred, _, _ = policy.step(observation.reshape(-1, *observation.shape))
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_true_rets = []
            ep_lens = []
        i = step % horizon
        observations[i] = observation
        vpreds[i] = vpred[0]
        actions[i] = action[0]
        prev_actions[i] = prevac

        clipped_action = action
        # Clip the actions to avoid out of bound error
        if isinstance(env.action_space, gym.spaces.Box):
            clipped_action = np.clip(action, env.action_space.low, env.action_space.high)

        if gail:
            rew = reward_giver.get_reward(observation, clipped_action[0])
            observation, true_rew, done, _info = env.step(clipped_action[0])
        else:
            observation, rew, done, _info = env.step(clipped_action[0])
            true_rew = rew
        rews[i] = rew
        true_rews[i] = true_rew
        dones[i] = done

        cur_ep_ret += rew
        cur_ep_true_ret += true_rew
        cur_ep_len += 1
        if done:
            ep_rets.append(cur_ep_ret)
            ep_true_rets.append(cur_ep_true_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_true_ret = 0
            cur_ep_len = 0
            if not isinstance(env, VecEnv):
                observation = env.reset()
        step += 1


def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)

    :param seg: (dict) the current segment of the trajectory (see traj_segment_generator return for more information)
    :param gamma: (float) Discount factor
    :param lam: (float) GAE factor
    """
    # last element is only used for last vtarg, but we already zeroed it if last new = 1
    new = np.append(seg["dones"], 0)
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    rew_len = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(rew_len, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for step in reversed(range(rew_len)):
        nonterminal = 1 - new[step + 1]
        delta = rew[step] + gamma * vpred[step + 1] * nonterminal - vpred[step]
        gaelam[step] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def flatten_lists(listoflists):
    """
    Flatten a python list of list

    :param listoflists: (list(list))
    :return: (list)
    """
    return [el for list_ in listoflists for el in list_]
