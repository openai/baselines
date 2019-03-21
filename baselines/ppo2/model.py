import functools

import tensorflow as tf
from baselines.common.tf_util import get_session, save_variables, load_variables
from baselines.common.tf_util import initialize

try:
    from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
    from mpi4py import MPI
    from baselines.common.mpi_util import sync_from_root
except ImportError:
    MPI = None


class Model(object):
    """
    We use this object to :
    __init__:
    - Creates the step_model
    - Creates the train_model

    train():
    - Make the training part (feedforward and retropropagation of gradients)

    save/load():
    - Save load the model
    """

    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                 nsteps, ent_coef, vf_coef, max_grad_norm,
                 name='ppo_model',
                 microbatch_size=None):
        self.sess = sess = get_session()
        self.name = name
        self.initial_state = None

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            with tf.name_scope('act_model'):
                # CREATE OUR TWO MODELS
                # act_model that is used for sampling
                act_model = policy(nbatch_act, 1, sess)

            with tf.name_scope('train_model'):
                # Train model for training
                if microbatch_size is None:
                    train_model = policy(nbatch_train, nsteps, sess)
                else:
                    train_model = policy(microbatch_size, nsteps, sess)

        with tf.variable_scope('losses'):
            # CREATE THE PLACEHOLDERS
            self.A = A = train_model.pdtype.sample_placeholder([None], name='action')
            self.ADV = ADV = tf.placeholder(tf.float32, [None], name='advantage')
            self.RETURNS = RETURNS = tf.placeholder(tf.float32, [None], name='reward')
            self.VALUE_PREV = VALUE_PREV = tf.placeholder(tf.float32, [None], name='value_prev')
            self.OLDNEGLOGPAC = OLDNEGLOGPAC = tf.placeholder(tf.float32, [None],
                                                              name='negative_log_p_action_old')
            self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [], name='clip_range')

            with tf.name_scope("neglogpac"):
                neglogpac = train_model.pd.neglogp(A)

            with tf.name_scope("entropy"):
                # Calculate the entropy
                # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
                entropy = tf.reduce_mean(train_model.pd.entropy())
                entropy_loss = (- ent_coef) * entropy

            with tf.name_scope("value_loss"):
                # CALCULATE THE LOSS
                value = train_model.value
                value_clipped = VALUE_PREV + tf.clip_by_value(value - VALUE_PREV, -CLIPRANGE, CLIPRANGE)
                vf_losses1 = tf.squared_difference(value, RETURNS)
                vf_losses2 = tf.squared_difference(value_clipped, RETURNS)
                vf_loss = 0.5 * vf_coef * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

            with tf.name_scope("policy_loss"):
                # Calculate ratio (pi current policy / pi old policy)
                ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
                pg_losses = -ADV * ratio
                pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
                pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))

            with tf.name_scope("approxkl"):
                approxkl = .5 * tf.reduce_mean(tf.squared_difference(neglogpac, OLDNEGLOGPAC))

            with tf.name_scope("clip_fraction"):
                clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

            with tf.name_scope("total_loss"):
                loss = pg_loss + entropy_loss + vf_loss

        with tf.name_scope('optimizer'):
            self.LR = LR = tf.placeholder(tf.float32, [], name='learning_rate')

            # UPDATE THE PARAMETERS USING LOSS
            # 1. Get the model parameters
            params = tf.trainable_variables(name)

            # 2. Build our trainer
            if MPI is not None:
                self.trainer = MpiAdamOptimizer(MPI.COMM_WORLD, learning_rate=LR, epsilon=1e-5)
            else:
                self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
            # 3. Calculate the gradients
            grads_and_var = self.trainer.compute_gradients(loss, params)
            grads, var = zip(*grads_and_var)

            if max_grad_norm is not None:
                # Clip the gradients (normalize)
                grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
            grads_and_var = list(zip(grads, var))

            self.grads = grads
            self.var = var
            self._train_op = self.trainer.apply_gradients(grads_and_var)

            self.loss_names = ['policy_loss', 'value_loss', 'entropy_loss', 'approxkl', 'clipfrac',
                               'total_loss']
            self.stats_list = [pg_loss, vf_loss, entropy_loss, approxkl, clipfrac, loss]

            self.train_model = train_model
            self.act_model = act_model

            self.save = functools.partial(save_variables, sess=sess)
            self.load = functools.partial(load_variables, sess=sess)

            initialize()
            sess.run(tf.initializers.local_variables())
            global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
            if MPI is not None:
                sync_from_root(sess, global_variables)  # pylint: disable=E1101

    def step_as_dict(self, **kwargs):
        return self.act_model.step(**kwargs)

    def step(self, observations, **kwargs):
        kwargs.update({'observations': observations})
        transition = self.act_model.step(**kwargs)
        states = transition['states'] if 'states' in transition else None
        return transition['actions'], transition['values'], states, transition['neglogpacs']

    def train(self,
              lr,
              cliprange,
              obs,
              advs,
              returns,
              actions,
              values,
              neglogpacs,
              **_kwargs):
        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        td_map = {
            self.train_model.X: obs,
            self.A: actions,
            self.ADV: advs,
            self.RETURNS: returns,
            self.LR: lr,
            self.CLIPRANGE: cliprange,
            self.OLDNEGLOGPAC: neglogpacs,
            self.VALUE_PREV: values,
        }

        td_map.update(self.train_model.feed_dict(**_kwargs))

        return self.sess.run(
            self.stats_list + [self._train_op],
            td_map
        )[:-1]
