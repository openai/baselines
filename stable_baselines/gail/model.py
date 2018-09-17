import gym

from stable_baselines.common import ActorCriticRLModel
from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.trpo_mpi import TRPO


class GAIL(ActorCriticRLModel):
    """
    Generative Adversarial Imitation Learning (GAIL)

    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) the discount value
    :param timesteps_per_batch: (int) the number of timesteps to run per batch (horizon)
    :param max_kl: (float) the kullback leiber loss threashold
    :param cg_iters: (int) the number of iterations for the conjugate gradient calculation
    :param lam: (float) GAE factor
    :param entcoeff: (float) the weight for the entropy loss
    :param cg_damping: (float) the compute gradient dampening factor
    :param vf_stepsize: (float) the value function stepsize
    :param vf_iters: (int) the value function's number iterations for learning
    :param pretrained_weight: (str) the save location for the pretrained weights
    :param hidden_size: ([int]) the hidden dimension for the MLP
    :param expert_dataset: (Dset) the dataset manager
    :param save_per_iter: (int) the number of iterations before saving
    :param checkpoint_dir: (str) the location for saving checkpoints
    :param g_step: (int) number of steps to train policy in each epoch
    :param d_step: (int) number of steps to train discriminator in each epoch
    :param task_name: (str) the name of the task (can be None)
    :param d_stepsize: (float) the reward giver stepsize
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    """

    def __init__(self, policy, env, pretrained_weight=False, hidden_size_adversary=100, adversary_entcoeff=1e-3,
                 expert_dataset=None, save_per_iter=1, checkpoint_dir="/tmp/gail/ckpt/", g_step=1, d_step=1,
                 task_name="task_name", d_stepsize=3e-4, verbose=0, _init_setup_model=True, **kwargs):
        super().__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=False,
                         _init_setup_model=_init_setup_model)

        self.trpo = TRPO(policy, env, verbose=verbose, _init_setup_model=False, **kwargs)
        self.trpo.using_gail = True
        self.trpo.pretrained_weight = pretrained_weight
        self.trpo.expert_dataset = expert_dataset
        self.trpo.save_per_iter = save_per_iter
        self.trpo.checkpoint_dir = checkpoint_dir
        self.trpo.g_step = g_step
        self.trpo.d_step = d_step
        self.trpo.task_name = task_name
        self.trpo.d_stepsize = d_stepsize
        self.trpo.hidden_size_adversary = hidden_size_adversary
        self.trpo.adversary_entcoeff = adversary_entcoeff

        if _init_setup_model:
            self.setup_model()

    def set_env(self, env):
        super().set_env(env)
        self.trpo.set_env(env)

    def setup_model(self):
        assert issubclass(self.policy, ActorCriticPolicy), "Error: the input policy for the GAIL model must be an " \
                                                           "instance of common.policies.ActorCriticPolicy."
        assert isinstance(self.action_space, gym.spaces.Box), "Error: GAIL requires a continuous action space."

        self.trpo.setup_model()

    def learn(self, total_timesteps, callback=None, seed=None, log_interval=100, tb_log_name="GAIL"):
        self.trpo.learn(total_timesteps, callback, seed, log_interval, tb_log_name)
        return self

    def predict(self, observation, state=None, mask=None, deterministic=False):
        return self.trpo.predict(observation, state, mask, deterministic=deterministic)

    def action_probability(self, observation, state=None, mask=None):
        return self.trpo.action_probability(observation, state, mask)

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
