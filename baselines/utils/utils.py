import numpy as np
# from osim.http.client import Client


def json_default(o):
    if isinstance(o, set):
        return list(o)
    return o.__dict__


class ClientToEnv(object):
    def __init__(self, client):
        """
        Reformats client environment to a local environment format.
        """
        self.reset = client.reset
        self.step = client.step


class DictToList(object):
    def __init__(self, env):
        """
        Formats Dictionary-type observation to List-type observation.
        """
        self.env = env

    def reset(self):
        state_desc = self.env.reset()
        return self._get_observation(state_desc)

    def step(self, action):
        state_desc, reward, done, info = self.env.step(action)
        return [self._get_observation(state_desc), reward, done, info]

    def _get_observation(self, state_desc):
        """
        Code from ProstheticsEnv.get_observation().

        https://github.com/stanfordnmbl/osim-rl/blob/master/osim/env/osim.py
        """
        # Augmented environment from the L2R challenge
        res = []
        pelvis = None

        for body_part in ["pelvis", "head","torso","toes_l","toes_r","talus_l","talus_r"]:
            if body_part in ["toes_r","talus_r"]:
                res += [0] * 9
                continue
            cur = []
            cur += state_desc["body_pos"][body_part][0:2]
            cur += state_desc["body_vel"][body_part][0:2]
            cur += state_desc["body_acc"][body_part][0:2]
            cur += state_desc["body_pos_rot"][body_part][2:]
            cur += state_desc["body_vel_rot"][body_part][2:]
            cur += state_desc["body_acc_rot"][body_part][2:]
            if body_part == "pelvis":
                pelvis = cur
                res += cur[1:]
            else:
                cur_upd = cur
                cur_upd[:2] = [cur[i] - pelvis[i] for i in range(2)]
                cur_upd[6:7] = [cur[i] - pelvis[i] for i in range(6,7)]
                res += cur

        for joint in ["ankle_l","ankle_r","back","hip_l","hip_r","knee_l","knee_r"]:
            res += state_desc["joint_pos"][joint]
            res += state_desc["joint_vel"][joint]
            res += state_desc["joint_acc"][joint]

        for muscle in sorted(state_desc["muscles"].keys()):
            res += [state_desc["muscles"][muscle]["activation"]]
            res += [state_desc["muscles"][muscle]["fiber_length"]]
            res += [state_desc["muscles"][muscle]["fiber_velocity"]]

        cm_pos = [state_desc["misc"]["mass_center_pos"][i] - pelvis[i] for i in range(2)]
        res = res + cm_pos + state_desc["misc"]["mass_center_vel"] + state_desc["misc"]["mass_center_acc"]

        return res


class JSONable(object):
    def __init__(self, env):
        """
        Converts NumPy ndarray type actions to list.
        """
        self.env = env
        self.reset = self.env.reset

    def step(self, action):
        if type(action) == np.ndarray:
            return self.env.step(action.tolist())
        else:
            return self.env.step(action)
