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
    def __init__(self, *, env, model, nsteps, gamma, gamma_int, int_coeff, ext_coeff, ob_space, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate for extrinsic rewardstrain_model.vf
        self.gamma = gamma
        # Discount rate for intrinsic rewards
        self.gamma_int = gamma_int
        # Intrinsic coefficent
        self.int_coeff = int_coeff 
        # Extrinsic coefficient
        self.ext_coeff = ext_coeff
        # Ob space
        self.ob_space = ob_space

    def init_obs_rnd_norm(self, obs_rnd_norm_nsteps, nenvs, ob_space):
        # This function is used to have initial normalization parameters by stepping
        # a random agent in the environment for a small nb of steps.
        print("Start to initialize the normalization parameters by stepping a random agent in the environment")

        all_obs = []
        for _ in range(obs_rnd_norm_nsteps * nenvs):
            actions, ext_values, int_values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
            all_obs.append(self.obs)
            
        ob_ = np.asarray(all_obs).astype(np.float32).reshape((-1, *ob_space.shape))
        self.model.rnd_model.rnd_ob_rms.update(ob_[:,:,:,-1:])
        print ("Initialization finished")

        all_obs.clear()
        

    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_ext_rewards, mb_int_rewards, mb_actions, mb_ext_values, mb_int_values, mb_combined_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[],[],[],[]
        mb_states = self.states

        epinfos = []
        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, ext_values, int_values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)

            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)

            mb_ext_values.append(ext_values)
            mb_int_values.append(int_values)

            combined_values = ext_values + int_values
            mb_combined_values.append(combined_values)
            
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs[:], ext_rewards, self.dones, infos = self.env.step(actions)
            mb_ext_rewards.append(ext_rewards)

            # Calculate intrinsic rewards
            int_rewards = self.model.rnd_model.calculate_intrinsic_rewards(self.obs[:,:,:,-1:])
            mb_int_rewards.append(int_rewards)

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_ext_rewards = np.asarray(mb_ext_rewards, dtype=np.float32)
        mb_int_rewards = np.asarray(mb_int_rewards, dtype=np.float32)

        mb_actions = np.asarray(mb_actions)
        mb_ext_values = np.asarray(mb_ext_values, dtype=np.float32)
        mb_int_values = np.asarray(mb_int_values, dtype=np.float32)
        mb_combined_values = np.asarray(mb_combined_values, dtype=np.float32)

        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_ext_values, last_int_values = self.model.values(self.obs, S=self.states, M=self.dones)

        # Update reward normalization parameters
        self.model.rnd_model.rnd_ir_rms.update(mb_int_rewards)
        
        # Normalize the intrinisic rewards
        mb_int_rewards = mb_int_rewards / np.sqrt(self.model.rnd_model.rnd_ir_rms.var)

        # Calculate returns and advantages for ER
        # discount/bootstrap off value fn
        mb_ext_returns = np.zeros_like(mb_ext_rewards)
        mb_advs_ext = np.zeros_like(mb_ext_rewards)

        #Calculate extrinsic returns and advantages.
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_ext_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_ext_values[t+1]
 
            delta = mb_ext_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_ext_values[t]
            mb_advs_ext[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_ext_returns = mb_advs_ext + mb_ext_values

        #Calculate intrinsic returns and advantages.
        mb_int_returns = np.zeros_like(mb_int_rewards)
        mb_advs_int = np.zeros_like(mb_int_rewards)

        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_int_values
            # No use dones for intrinisc reward
            else:
                nextnonterminal = 0.0

            delta = mb_int_rewards[t] + self.gamma_int * nextvalues * nextnonterminal - mb_int_values[t]
            mb_advs_int[t] = lastgaelam = delta + self.gamma_int * self.lam * nextnonterminal * lastgaelam
        mb_int_returns = mb_advs_int + mb_int_values

        #Combine the extrinsic and intrinsic advantages.
        mb_combined_returns = self.int_coeff * mb_int_returns + self.ext_coeff * mb_ext_returns

        # Update norm parameters after the rollout is completed
        obs_ = mb_obs.reshape((-1, *self.ob_space.shape))
        self.model.rnd_model.rnd_ob_rms.update(obs_[:,:,:,-1:])

        return (*map(sf01, (mb_obs, mb_combined_returns, mb_int_returns, mb_ext_returns, mb_dones, mb_actions, mb_ext_values, mb_int_values, mb_combined_values, mb_neglogpacs)),
            mb_states, epinfos)

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


