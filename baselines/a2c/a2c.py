import time
import tensorflow as tf

from baselines import logger

from baselines.common import set_global_seeds, explained_variance
from baselines.common.models import get_network_builder
from baselines.common.policies import PolicyWithValue

from baselines.a2c.utils import InverseLinearTimeDecay
from baselines.a2c.runner import Runner
from baselines.ppo2.ppo2 import safemean
import os.path as osp
from collections import deque

class Model(tf.keras.Model):

    """
    We use this class to :
        __init__:
        - Creates the step_model
        - Creates the train_model

        train():
        - Make the training part (feedforward and retropropagation of gradients)

        save/load():
        - Save load the model
    """
    def __init__(self, *, ac_space, policy_network, nupdates,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6)):

        super(Model, self).__init__(name='A2CModel')
        self.train_model = PolicyWithValue(ac_space, policy_network, value_network=None, estimate_q=False)
        lr_schedule = InverseLinearTimeDecay(initial_learning_rate=lr, nupdates=nupdates)
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule, rho=alpha, epsilon=epsilon)

        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.step = self.train_model.step
        self.value = self.train_model.value
        self.initial_state = self.train_model.initial_state

    @tf.function
    def train(self, obs, states, rewards, masks, actions, values):
        advs = rewards - values
        with tf.GradientTape() as tape:
            policy_latent = self.train_model.policy_network(obs)
            pd, _ = self.train_model.pdtype.pdfromlatent(policy_latent)
            neglogpac = pd.neglogp(actions)
            entropy = tf.reduce_mean(pd.entropy())
            vpred = self.train_model.value(obs)
            vf_loss = tf.reduce_mean(tf.square(vpred - rewards))
            pg_loss = tf.reduce_mean(advs * neglogpac)
            loss = pg_loss - entropy * self.ent_coef + vf_loss * self.vf_coef

        var_list = tape.watched_variables()
        grads = tape.gradient(loss, var_list)
        grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        grads_and_vars = list(zip(grads, var_list))
        self.optimizer.apply_gradients(grads_and_vars)

        return pg_loss, vf_loss, entropy


def learn(
    network,
    env,
    seed=None,
    nsteps=5,
    total_timesteps=int(80e6),
    vf_coef=0.5,
    ent_coef=0.01,
    max_grad_norm=0.5,
    lr=7e-4,
    lrschedule='linear',
    epsilon=1e-5,
    alpha=0.99,
    gamma=0.99,
    log_interval=100,
    load_path=None,
    **network_kwargs):

    '''
    Main entrypoint for A2C algorithm. Train a policy with given network architecture on a given environment using a2c algorithm.

    Parameters:
    -----------

    network:            policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                        specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                        tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                        neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                        See baselines.common/policies.py/lstm for more details on using recurrent nets in policies


    env:                RL environment. Should implement interface similar to VecEnv (baselines.common/vec_env) or be wrapped with DummyVecEnv (baselines.common/vec_env/dummy_vec_env.py)


    seed:               seed to make random number sequence in the alorightm reproducible. By default is None which means seed from system noise generator (not reproducible)

    nsteps:             int, number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                        nenv is number of environment copies simulated in parallel)

    total_timesteps:    int, total number of timesteps to train on (default: 80M)

    vf_coef:            float, coefficient in front of value function loss in the total loss function (default: 0.5)

    ent_coef:           float, coeffictiant in front of the policy entropy in the total loss function (default: 0.01)

    max_gradient_norm:  float, gradient is clipped to have global L2 norm no more than this value (default: 0.5)

    lr:                 float, learning rate for RMSProp (current implementation has RMSProp hardcoded in) (default: 7e-4)

    lrschedule:         schedule of learning rate. Can be 'linear', 'constant', or a function [0..1] -> [0..1] that takes fraction of the training progress as input and
                        returns fraction of the learning rate (specified as lr) as output

    epsilon:            float, RMSProp epsilon (stabilizes square root computation in denominator of RMSProp update) (default: 1e-5)

    alpha:              float, RMSProp decay parameter (default: 0.99)

    gamma:              float, reward discounting parameter (default: 0.99)

    log_interval:       int, specifies how frequently the logs are printed out (default: 100)

    **network_kwargs:   keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                        For instance, 'mlp' network architecture has arguments num_hidden and num_layers.

    '''



    set_global_seeds(seed)

    total_timesteps = int(total_timesteps)

    # Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    if isinstance(network, str):
        network_type = network
        policy_network_fn = get_network_builder(network_type)(**network_kwargs)
        policy_network = policy_network_fn(ob_space.shape)

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nupdates = total_timesteps // nbatch

    # Instantiate the model object (that creates step_model and train_model)
    model = Model(ac_space=ac_space, policy_network=policy_network, nupdates=nupdates, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps)

    if load_path is not None:
        load_path = osp.expanduser(load_path)
        ckpt = tf.train.Checkpoint(model=model)
        manager = tf.train.CheckpointManager(ckpt, load_path, max_to_keep=None)
        ckpt.restore(manager.latest_checkpoint)

    # Instantiate the runner object
    runner = Runner(env, model, nsteps=nsteps, gamma=gamma)
    epinfobuf = deque(maxlen=100)

    # Start total timer
    tstart = time.time()

    for update in range(1, nupdates+1):
        # Get mini batch of experiences
        obs, states, rewards, masks, actions, values, epinfos = runner.run()
        epinfobuf.extend(epinfos)

        obs = tf.constant(obs)
        if states is not None:
            states = tf.constant(states)
        rewards = tf.constant(rewards)
        masks = tf.constant(masks)
        actions = tf.constant(actions)
        values = tf.constant(values)
        policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)
        nseconds = time.time()-tstart

        # Calculate the fps (frame per second)
        fps = int((update*nbatch)/nseconds)
        if update % log_interval == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.record_tabular("eprewmean", safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.record_tabular("eplenmean", safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.dump_tabular()
    return model

