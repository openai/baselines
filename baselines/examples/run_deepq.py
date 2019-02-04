# import tensorflow as tf
# import random

# from baselines import deepq
# from baselines.common.identity_env import IdentityEnv

# def test_identity_deeqp():

#     with tf.Graph().as_default():
#         env = IdentityEnv(10)
#         random.seed(0)

#         tf.set_random_seed(0)

#         param_noise = False
#         model = deepq.models.mlp([32])
#         act = deepq.learn(
#             env,
#             q_func=model,
#             lr=1e-3,
#             max_timesteps=10000,
#             buffer_size=50000,
#             exploration_fraction=0.1,
#             exploration_final_eps=0.02,
#             print_freq=10,
#             param_noise=param_noise,
#         )

#         tf.set_random_seed(0)

#         N_TRIALS = 1000
#         sum_rew = 0
#         obs = env.reset()
#         for i in range(N_TRIALS):
#             obs, rew, done, _ = env.step(act([obs]))
#             sum_rew += rew

#         assert sum_rew > 0.9 * N_TRIALS

# if __name__ == '__main__':
#     test_identity()

import os
import zipfile
import tempfile
import cloudpickle
import numpy as np
import tensorflow as tf
from utils import logger
from algos.deepq import DeepDQN
from utils.schedules import LinearSchedule
from utils.inputs import observation_input
from dstruct.memory.buffers import ReplayBuffer, PrioritizedReplayBuffer
# from baselines.common.input import observation_input

# ================================================================
# Placeholders
# ================================================================


class TfInput(object):
    def __init__(self, name="(unnamed)"):
        """
        Generalized Tensorflow placeholder. The main differences are:
            - possibly uses multiple placeholders internally and
              returns multiple values
            - can apply light postprocessing to the value feed to placeholder.
        """
        self.name = name

    def get(self):
        """
        Return the tf variable(s) representing the possibly
        postprocessed value of placeholder(s).
        """
        raise NotImplementedError()

    def make_feed_dict(data):
        """Given data input it to the placeholder(s)."""
        raise NotImplementedError()


class PlaceholderTfInput(TfInput):
    def __init__(self, placeholder):
        """Wrapper for regular tensorflow placeholder."""
        super().__init__(placeholder.name)
        self._placeholder = placeholder

    def get(self):
        return self._placeholder

    def make_feed_dict(self, data):
        return {self._placeholder: data}


class Uint8Input(PlaceholderTfInput):
    def __init__(self, shape, name=None):
        """
        Takes input in uint8 format which is cast to float32 and
        divided by 255 before passing it to the model.

        On GPU this ensures lower data transfer times.

        Parameters
        ----------
        shape: [int]
            shape of the tensor.
        name: str
            name of the underlying placeholder
        """

        super().__init__(tf.placeholder(tf.uint8, [None] + list(shape), name=name))
        self._shape = shape
        self._output = tf.cast(super().get(), tf.float32) / 255.0

    def get(self):
        return self._output


class ObservationInput(PlaceholderTfInput):
    def __init__(self, observation_space, name=None):
        """Creates an input placeholder tailored to a specific observation space

        Parameters
        ----------

        observation_space:
                observation space of the environment. Should be one of
                the gym.spaces types
        name: str
                tensorflow name of the underlying placeholder
        """
        inpt, self.processed_inpt = observation_input(observation_space, name=name)
        super().__init__(inpt)

    def get(self):
        return self.processed_inpt


class ActWrapper(DeepDQN):
    def __init__(self, act, act_params):
        super(ActWrapper, self).__init__()
        self._act = act
        self._act_params = act_params

    @staticmethod
    def load(self, path):
        with open(path, "rb") as f:
            model_data, act_params = cloudpickle.load(f)
        act = self.build_act(**act_params)
        sess = self.init_session().__enter__()
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(
                arc_path,
                'r',
                zipfile.ZIP_DEFLATED
            ).extractall(td)

            self.load_state(os.path.join(td, "model"))

        sess.close()
        del sess
        return ActWrapper(act, act_params)

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def save(self, path=None):
        """Save model to a pickle located at `path`"""
        if path is None:
            path = os.path.join(logger.get_dir(), "model.pkl")

        with tempfile.TemporaryDirectory() as td:
            self.save_state(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(
                                file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()

        with open(path, "wb") as f:
            cloudpickle.dump((model_data, self._act_params), f)


def load(path):
    """Load act function that was returned by learn function.

    Parameters
    ----------
    path: str
        path to the act function pickle

    Returns
    -------
    act: ActWrapper
        function that takes a batch of observations
        and returns actions.
    """
    return ActWrapper.load(path)


def fit(
        env,
        q_func,
        lr=5e-4,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        train_freq=1,
        batch_size=32,
        print_freq=100,
        checkpoint_freq=10000,
        checkpoint_path=None,
        learning_starts=1000,
        gamma=1.0,
        target_network_update_freq=500,
        prioritized_replay=False,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta0=0.4,
        prioritized_replay_beta_iters=None,
        prioritized_replay_eps=1e-6,
        param_noise=False,
        callback=None
):
    """Train a deepq model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    lr: float
        learning rate for adam optimizer
    max_timesteps: int
        number of env steps to optimizer for
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration
        rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
        set to None to disable printing
    batch_size: int
        size of a batched sampled from replay buffer for training
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version
        is restored at the end of the training. If you do not wish to
        restore the best version at the end of the training set this
        variable to None.
    learning_starts: int
        how many steps of the model to collect transitions for before
        learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    prioritized_replay: True
        if True prioritized replay buffer will be used.
    prioritized_replay_alpha: float
        alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        number of iterations over which beta will be annealed from
        initial value to 1.0. If set to None equals to max_timesteps.
    prioritized_replay_eps: float
        epsilon to add to the TD errors when updating priorities.
    callback: (locals, globals) -> None
        function called at every steps with state of the algorithm.
        If callback returns true training stops.

    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load
        it.  See header of baselines/deepq/categorical.py for details
        on the act function.
    """
    # Create all the functions necessary to train the model

    model = DeepDQN()
    sess = model.init_session().__enter__()

    # capture the shape outside the closure so that the env object is
    # not serialized by cloudpickle when serializing make_obs_ph

    def make_obs_ph(name):
        return ObservationInput(env.observation_space, name=name)

    act, train, update_target, debug = model.build_train(
        make_obs_ph,
        q_func,
        env.action_space.n,
        tf.train.AdamOptimizer(learning_rate=lr),
        10,
        gamma,
        param_noise
    )

    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'num_actions': env.action_space.n,
    }

    act = ActWrapper(act, act_params)

    # Create the replay buffer
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(
            buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = max_timesteps
        beta_schedule = LinearSchedule(
            prioritized_replay_beta_iters,
            initial_p=prioritized_replay_beta0,
            final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(
        schedule_timesteps=int(exploration_fraction * max_timesteps),
        initial_p=1.0,
        final_p=exploration_final_eps)

    # Initialize the parameters and copy them to the target network.
    model.init_vars()
    update_target()

    episode_rewards = [0.0]
    saved_mean_reward = None
    obs = env.reset()
    reset = True

    with tempfile.TemporaryDirectory() as td:
        td = checkpoint_path or td

        model_file = os.path.join(td, "model")
        model_saved = False
        if tf.train.latest_checkpoint(td) is not None:
            model.load_state(model_file)
            logger.log('Loaded model from {}'.format(model_file))
            model_saved = True

        for t in range(max_timesteps):
            if callback is not None:
                if callback(locals(), globals()):
                    break
            # Take action and update exploration to the newest value
            kwargs = {}
            if not param_noise:
                update_eps = exploration.value(t)
                update_param_noise_threshold = 0.
            else:
                update_eps = 0.
                # Compute the threshold such that the KL divergence
                # between perturbed and non-perturbed
                # policy is comparable to eps-greedy exploration with
                # eps = exploration.value(t).  See Appendix C.1 in
                # Parameter Space Noise for Exploration, Plappert et
                # al., 2017
                # for detailed explanation.
                update_param_noise_threshold = -np.log(1. - exploration.value(
                    t) + exploration.value(t) / float(env.action_space.n))
                kwargs['reset'] = reset
                kwargs['update_param_noise_threshold'] = \
                    update_param_noise_threshold
                kwargs['update_param_noise_scale'] = True
            action = act(
                np.array(obs)[None], update_eps=update_eps, **kwargs
            )[0]
            env_action = action
            reset = False
            new_obs, rew, done, _ = env.step(env_action)
            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = env.reset()
                episode_rewards.append(0.0)
                reset = True

            if t > learning_starts and t % train_freq == 0:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if prioritized_replay:
                    experience = replay_buffer.sample(
                        batch_size, beta=beta_schedule.value(t)
                    )
                    (obses_t, actions, rewards, obses_tp1, dones, weights,
                     batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = \
                        replay_buffer.sample(batch_size)
                    weights, batch_idxes = np.ones_like(rewards), None
                td_errors = train(
                    obses_t,
                    actions,
                    rewards,
                    obses_tp1,
                    dones,
                    weights
                )
                if prioritized_replay:
                    new_priorities = np.abs(td_errors) + prioritized_replay_eps
                    replay_buffer.update_priorities(
                        batch_idxes,
                        new_priorities
                    )

            if t > learning_starts and t % target_network_update_freq == 0:
                # Update target network periodically.
                update_target()

            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            num_episodes = len(episode_rewards)
            if done and print_freq is not None and len(
                    episode_rewards) % print_freq == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean 100 episode reward",
                                      mean_100ep_reward)
                logger.record_tabular("% time spent exploring",
                                      int(100 * exploration.value(t)))
                logger.dump_tabular()

            if (checkpoint_freq is not None and t > learning_starts
                    and num_episodes > 100 and t % checkpoint_freq == 0):
                if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                    if print_freq is not None:
                        logger.log(
                            "Saving model due to mean reward increase: {} -> {}".
                            format(saved_mean_reward, mean_100ep_reward)
                        )
                    model.save_state(model_file)
                    model_saved = True
                    saved_mean_reward = mean_100ep_reward
        if model_saved:
            if print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(
                    saved_mean_reward)
                )
            model.load_state(model_file)

    return act


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID',
                        default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--prioritized', type=int, default=1)
    parser.add_argument('--prioritized-replay-alpha', type=float, default=0.6)
    parser.add_argument('--dueling', type=int, default=1)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    parser.add_argument('--checkpoint-freq', type=int, default=10000)
    parser.add_argument('--checkpoint-path', type=str, default=None)

    args = parser.parse_args()
    logger.configure()
    set_global_seeds(args.seed)
    env = make_atari(args.env)
    env = Monitor(env, logger.get_dir())
    env = wrap_deepmind(env)
    model = cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=bool(args.dueling),
    )

    fit(
        env,
        q_func=model,
        lr=1e-4,
        max_timesteps=args.num_timesteps,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=bool(args.prioritized),
        prioritized_replay_alpha=args.prioritized_replay_alpha,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_path=args.checkpoint_path,
    )

    env.close()
    sess = tf.get_default_session()
    del sess


if __name__ == '__main__':
    import argparse
    from bench import Monitor
    from algos.deepq import cnn_to_mlp
    from utils.misc import set_global_seeds
    from common.vec_env.atari_wrappers import make_atari, wrap_deepmind

    main()
