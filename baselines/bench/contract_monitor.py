import time

from baselines.bench.monitor import Monitor, ResultsWriter


class ContractMonitor(Monitor):
    def __init__(self,
                 env,
                 filename,
                 contract_set,
                 allow_early_resets=False,
                 reset_keywords=(),
                 info_keywords=()):
        super(ContractMonitor, self).__init__(
            env, filename, allow_early_resets, reset_keywords, info_keywords)
        self.results_writer = ResultsWriter(
            filename,
            header={"t_start": time.time(), 'env_id' : env.spec and env.spec.id},
            extra_keys=reset_keywords + info_keywords + ('rmod',) + tuple(contract_set.get_contract_names())
        )
        self.contract_set = contract_set
        self.violations = []
        self.reward_modifications = []
        self.episode_violations = []
        self.episode_rewards_mods = []

    def reset(self, **kwargs):
        super(ContractMonitor, self).reset(**kwargs)
        self.contract_set.reset()
        self.violations = []
        self.reward_modifications = []

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        ob, rew, done, info = self.env.step(action)
        viols = self.contract_set.step(action)
        rew_mod = [
            int(v) * r
            for (v, r) in zip(viols, self.contract_set.get_violation_rewards())
        ]
        self.update(ob, rew, done, info, viols, rew_mod)
        return (ob, rew + sum(rew_mod), done, info)

    def update(self, ob, rew, done, info, viol, rew_mod):
        self.rewards.append(rew)
        self.violations.append(viol)
        self.reward_modifications.append(sum(rew_mod))
        if done:
            ep_viols = [sum(cv) for cv in zip(*self.violations)]
            ep_rew_mods = sum(self.reward_modifications)

            self.needs_reset = True
            eprew = sum(self.rewards)
            eprewmod = sum(self.reward_modifications)
            eplen = len(self.rewards)
            ep_viols = [sum(cv) for cv in zip(*self.violations)]
            ep_viol_dict = dict(
                zip(self.contract_set.get_contract_names(), ep_viols))
            epinfo = {
                "r": round(eprew, 6),
                "rmod": eprewmod,
                "l": eplen,
                "t": round(time.time() - self.tstart, 6)
            }
            epinfo = dict(list(epinfo.items()) + list(ep_viol_dict.items()))
            for k in self.info_keywords:
                epinfo[k] = info[k]
            self.episode_rewards.append(eprew)
            self.episode_lengths.append(eplen)
            self.episode_times.append(time.time() - self.tstart)
            self.episode_violations.append(ep_viols)
            self.episode_rewards_mods.append(ep_rew_mods)
            epinfo.update(self.current_reset_info)
            self.results_writer.write_row(epinfo)

            if isinstance(info, dict):
                info['episode'] = epinfo

        self.total_steps += 1

    def get_episode_violations(self):
        return self.episode_violations

    def get_episode_reward_modifications(self):
        return self.episode_rewards_mods