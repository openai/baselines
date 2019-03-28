from stable_baselines.common import ActorCriticRLModel
from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.trpo_mpi import TRPO


class GAIL(ActorCriticRLModel):
    """
    Generative Adversarial Imitation Learning (GAIL)

    .. warning::

        Images are not yet handled properly by the current implementation


    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param expert_dataset: (ExpertDataset) the dataset manager
    :param gamma: (float) the discount value
    :param timesteps_per_batch: (int) the number of timesteps to run per batch (horizon)
    :param max_kl: (float) the kullback leiber loss threashold
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
        super().__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=False,
                         _init_setup_model=_init_setup_model)

        self.trpo = TRPO(policy, env, verbose=verbose, _init_setup_model=False, **kwargs)
        self.trpo.using_gail = True
        self.trpo.expert_dataset = expert_dataset
        self.trpo.g_step = g_step
        self.trpo.d_step = d_step
        self.trpo.d_stepsize = d_stepsize
        self.trpo.hidden_size_adversary = hidden_size_adversary
        self.trpo.adversary_entcoeff = adversary_entcoeff
        self.env = self.trpo.env

        if _init_setup_model:
            self.setup_model()

    def _get_pretrain_placeholders(self):
        pass

    def pretrain(self, dataset, n_epochs=10, learning_rate=1e-4,
                 adam_epsilon=1e-8, val_interval=None):
        self.trpo.pretrain(dataset, n_epochs=n_epochs, learning_rate=learning_rate,
                           adam_epsilon=adam_epsilon, val_interval=val_interval)
        return self

    def set_env(self, env):
        self.trpo.set_env(env)
        self.env = self.trpo.env

    def setup_model(self):
        assert issubclass(self.policy, ActorCriticPolicy), "Error: the input policy for the GAIL model must be an " \
                                                           "instance of common.policies.ActorCriticPolicy."
        self.trpo.setup_model()

    def learn(self, total_timesteps, callback=None, seed=None, log_interval=100, tb_log_name="GAIL",
              reset_num_timesteps=True):
        assert self.trpo.expert_dataset is not None, "You must pass an expert dataset to GAIL for training"
        self.trpo.learn(total_timesteps, callback, seed, log_interval, tb_log_name, reset_num_timesteps)
        return self

    def predict(self, observation, state=None, mask=None, deterministic=False):
        return self.trpo.predict(observation, state=state, mask=mask, deterministic=deterministic)

    def action_probability(self, observation, state=None, mask=None, actions=None):
        return self.trpo.action_probability(observation, state=state, mask=mask, actions=actions)

    def save(self, save_path):
        self.trpo.save(save_path)

    @classmethod
    def load(cls, load_path, env=None, **kwargs):
        data, params = cls._load_from_file(load_path)

        model = cls(policy=data["policy"], env=None, _init_setup_model=False)
        model.trpo.__dict__.update(data)
        model.trpo.__dict__.update(kwargs)
        model.set_env(env)
        model.setup_model()

        restores = []
        for param, loaded_p in zip(model.trpo.params, params):
            restores.append(param.assign(loaded_p))
        model.trpo.sess.run(restores)

        return model
