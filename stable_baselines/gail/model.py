from stable_baselines.trpo_mpi import TRPO


class GAIL(TRPO):
    """
    Generative Adversarial Imitation Learning (GAIL)

    .. warning::

        Images are not yet handled properly by the current implementation


    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param expert_dataset: (ExpertDataset) the dataset manager
    :param gamma: (float) the discount value
    :param timesteps_per_batch: (int) the number of timesteps to run per batch (horizon)
    :param max_kl: (float) the Kullback-Leibler loss threshold
    :param cg_iters: (int) the number of iterations for the conjugate gradient calculation
    :param lam: (float) GAE factor
    :param entcoeff: (float) the weight for the entropy loss
    :param cg_damping: (float) the compute gradient dampening factor
    :param vf_stepsize: (float) the value function stepsize
    :param vf_iters: (int) the value function's number iterations for learning
    :param hidden_size: ([int]) the hidden dimension for the MLP
    :param g_step: (int) number of steps to train policy in each epoch
    :param d_step: (int) number of steps to train discriminator in each epoch
    :param d_stepsize: (float) the reward giver stepsize
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    """

    def __init__(self, policy, env, expert_dataset=None,
                 hidden_size_adversary=100, adversary_entcoeff=1e-3,
                 g_step=3, d_step=1, d_stepsize=3e-4, verbose=0,
                 _init_setup_model=True, **kwargs):
        super().__init__(policy, env, verbose=verbose, _init_setup_model=False, **kwargs)
        self.using_gail = True
        self.expert_dataset = expert_dataset
        self.g_step = g_step
        self.d_step = d_step
        self.d_stepsize = d_stepsize
        self.hidden_size_adversary = hidden_size_adversary
        self.adversary_entcoeff = adversary_entcoeff

        if _init_setup_model:
            self.setup_model()

    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="GAIL",
              reset_num_timesteps=True):
        assert self.expert_dataset is not None, "You must pass an expert dataset to GAIL for training"
        return super().learn(total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps)
