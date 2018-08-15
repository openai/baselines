import joblib
import tensorflow as tf
from policies.model import Model
from utils.schedules import Scheduler
from optimizers.kfac import KfacOptimizer


class AcktrDiscrete(Model):

    def __init__(
            self,
            policy,
            ob_space,
            ac_space,
            nenvs,
            total_timesteps,
            nsteps=20,
            ent_coef=0.01,
            vf_coef=0.5,
            vf_fisher_coef=1.0,
            lr=0.25,
            max_grad_norm=0.5,
            kfac_clip=0.001,
            lrschedule='linear'
    ):
        self.sess = sess = self.init_session()
        nact = ac_space.n
        nbatch = nenvs * nsteps
        A = tf.placeholder(tf.int32, [nbatch])
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        PG_LR = tf.placeholder(tf.float32, [])
        VF_LR = tf.placeholder(tf.float32, [])

        self.model = step_model = policy(sess,ob_space, ac_space, nenvs, 1,
                                         reuse=False)
        self.model2 = train_model = policy(sess,ob_space, ac_space,
                                           nenvs*nsteps, nsteps, reuse=True)

        logpac = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=train_model.pi, labels=A
        )
        self.logits = logits = train_model.pi

        ##training loss
        pg_loss = tf.reduce_mean(ADV*logpac)
        entropy = tf.reduce_mean(Model().categorical_entropy(train_model.pi))
        pg_loss = pg_loss - ent_coef * entropy
        vf_loss = tf.reduce_mean(
            tf.losses.mean_squared_error(tf.squeeze(train_model.vf), R)
        )
        train_loss = pg_loss + vf_coef * vf_loss


        ##Fisher loss construction
        self.pg_fisher = pg_fisher_loss = -tf.reduce_mean(logpac)
        sample_net = train_model.vf + tf.random_normal(tf.shape(train_model.vf))
        self.vf_fisher = vf_fisher_loss = - vf_fisher_coef*tf.reduce_mean(
            tf.pow(train_model.vf - tf.stop_gradient(sample_net), 2)
        )
        self.joint_fisher = joint_fisher_loss = pg_fisher_loss + vf_fisher_loss

        # self.params=params = find_trainable_variables("model")
        self.params=params = tf.trainable_variables()

        self.grads_check = grads = tf.gradients(train_loss, params)

        with tf.device('/gpu:0'):
            self.optim = optim = KfacOptimizer(
                learning_rate=PG_LR,
                clip_kl=kfac_clip,
                momentum=0.9,
                kfac_update=1,
                epsilon=0.01,
                stats_decay=0.99,
                async=1,
                cold_iter=10,
                max_grad_norm=max_grad_norm
            )

            update_stats_op = optim.compute_and_apply_stats(
                joint_fisher_loss,
                var_list=params
            )
            train_op, q_runner = optim.apply_gradients(list(zip(grads,params)))
        self.q_runner = q_runner
        self.lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values):
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = self.lr.value()

            td_map = {
                train_model.X: obs,
                A: actions,
                ADV: advs,
                R: rewards,
                PG_LR: cur_lr
            }
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks

            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, train_op],
                td_map
            )
            return policy_loss, value_loss, policy_entropy

        def save(save_path):
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)

        self.train = train
        self.save = save
        self.load = load
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        tf.global_variables_initializer().run(session=sess)
