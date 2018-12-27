import tensorflow as tf
import numpy as np
from baselines.ppo2.model import Model

class MicrobatchedModel(Model):
    """
    Model that does training one microbatch at a time - when gradient computation
    on the entire minibatch causes some overflow
    """
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm, microbatch_size):

        self.nmicrobatches = nbatch_train // microbatch_size
        self.microbatch_size = microbatch_size
        assert nbatch_train % microbatch_size == 0, 'microbatch_size ({}) should divide nbatch_train ({}) evenly'.format(microbatch_size, nbatch_train)

        super().__init__(
                policy=policy,
                ob_space=ob_space,
                ac_space=ac_space,
                nbatch_act=nbatch_act,
                nbatch_train=microbatch_size,
                nsteps=nsteps,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm)

        self.grads_ph = [tf.placeholder(dtype=g.dtype, shape=g.shape) for g in self.grads]
        grads_ph_and_vars = list(zip(self.grads_ph, self.var))
        self._apply_gradients_op = self.trainer.apply_gradients(grads_ph_and_vars)


    def train(self, lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
        assert states is None, "microbatches with recurrent models are not supported yet"

        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        # Initialize empty list for per-microbatch stats like pg_loss, vf_loss, entropy, approxkl (whatever is in self.stats_list)
        stats_vs = []

        for microbatch_idx in range(self.nmicrobatches):
            _sli = range(microbatch_idx * self.microbatch_size, (microbatch_idx+1) * self.microbatch_size)
            td_map = {
                self.train_model.X: obs[_sli],
                self.A:actions[_sli],
                self.ADV:advs[_sli],
                self.R:returns[_sli],
                self.CLIPRANGE:cliprange,
                self.OLDNEGLOGPAC:neglogpacs[_sli],
                self.OLDVPRED:values[_sli]
            }

            # Compute gradient on a microbatch (note that variables do not change here) ...
            grad_v, stats_v  = self.sess.run([self.grads, self.stats_list], td_map)
            if microbatch_idx == 0:
                sum_grad_v = grad_v
            else:
                # .. and add to the total of the gradients
                for i, g in enumerate(grad_v):
                    sum_grad_v[i] += g
            stats_vs.append(stats_v)

        feed_dict = {ph: sum_g / self.nmicrobatches for ph, sum_g in zip(self.grads_ph, sum_grad_v)}
        feed_dict[self.LR] = lr
        # Update variables using average of the gradients
        self.sess.run(self._apply_gradients_op, feed_dict)
        # Return average of the stats
        return np.mean(np.array(stats_vs), axis=0).tolist()



