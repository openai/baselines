import numpy as np

from baselines.common.runners import AbstractEnvRunner


class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """

    def __init__(self, *, env, model, nsteps, gamma, ob_space, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)

        self.lam = lam  # Lambda used in GAE (General Advantage Estimation)
        self.gamma = gamma  # Discount rate for rewards
        self.ob_space = ob_space

    def run(self):
        minibatch = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "values": [],
            "dones": [],
            "neglogpacs": [],
        }

        data_type = {
            "observations": self.obs.dtype,
            "actions": np.float32,
            "rewards": np.float32,
            "values": np.float32,
            "dones": np.float32,
            "neglogpacs": np.float32,
        }

        prev_transition = {'next_states': self.model.initial_state} if self.model.initial_state is not None else {}
        epinfos = []

        # For n in range number of steps
        for _ in range(self.nsteps):
            transitions = {}
            transitions['observations'] = self.obs.copy()
            transitions['dones'] = self.dones
            if 'next_states' in prev_transition:
                transitions['states'] = prev_transition['next_states']
            transitions.update(self.model.step_as_dict(**transitions))

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs, transitions['rewards'], self.dones, infos = self.env.step(transitions['actions'])
            self.dones = np.array(self.dones, dtype=np.float)

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    epinfos.append(maybeepinfo)

            for key in transitions:
                if key not in minibatch:
                    minibatch[key] = []
                minibatch[key].append(transitions[key])
            prev_transition = transitions

        for key in minibatch:
            dtype = data_type[key] if key in data_type else np.float
            minibatch[key] = np.array(minibatch[key], dtype=dtype)

        transitions['observations'] = self.obs.copy()
        transitions['dones'] = self.dones
        if 'states' in transitions:
            transitions['states'] = transitions.pop('next_states')

        for key in minibatch:
            dtype = data_type[key] if key in data_type else np.float
            minibatch[key] = np.asarray(minibatch[key], dtype=dtype)

        last_values = self.model.step_as_dict(**transitions)['values']

        # Calculate returns and advantages.
        minibatch['advs'], minibatch['returns'] = \
            self.advantage_and_returns(values=minibatch['values'],
                                       rewards=minibatch['rewards'],
                                       dones=minibatch['dones'],
                                       last_values=last_values,
                                       last_dones=self.dones,
                                       gamma=self.gamma)

        for key in minibatch:
            minibatch[key] = sf01(minibatch[key])

        minibatch['epinfos'] = epinfos
        return minibatch

    def advantage_and_returns(self, values, rewards, dones, last_values, last_dones, gamma,
                              use_non_episodic_rewards=False):
        """
        calculate Generalized Advantage Estimation (GAE), https://arxiv.org/abs/1506.02438
        see also Proximal Policy Optimization Algorithms, https://arxiv.org/abs/1707.06347
        """

        advantages = np.zeros_like(rewards)
        lastgaelam = 0  # Lambda used in General Advantage Estimation
        for t in reversed(range(self.nsteps)):
            if not use_non_episodic_rewards:
                if t == self.nsteps - 1:
                    next_non_terminal = 1.0 - last_dones
                else:
                    next_non_terminal = 1.0 - dones[t + 1]
            else:
                next_non_terminal = 1.0
            next_value = values[t + 1] if t < self.nsteps - 1 else last_values
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            advantages[t] = lastgaelam = delta + gamma * self.lam * next_non_terminal * lastgaelam
        returns = advantages + values
        return advantages, returns


def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
