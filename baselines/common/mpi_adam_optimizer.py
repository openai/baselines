import numpy as np
import tensorflow as tf
from mpi4py import MPI

class MpiAdamOptimizer(tf.train.AdamOptimizer):
    """Adam optimizer that averages gradients across mpi processes."""
    def __init__(self, comm, **kwargs):
        self.comm = comm
        tf.train.AdamOptimizer.__init__(self, **kwargs)
    def compute_gradients(self, loss, var_list, **kwargs):
        grads_and_vars = tf.train.AdamOptimizer.compute_gradients(self, loss, var_list, **kwargs)
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        flat_grad = tf.concat([tf.reshape(g, (-1,)) for g, v in grads_and_vars], axis=0)
        shapes = [v.shape.as_list() for g, v in grads_and_vars]
        sizes = [int(np.prod(s)) for s in shapes]

        num_tasks = self.comm.Get_size()
        buf = np.zeros(sum(sizes), np.float32)

        sess = tf.get_default_session()
        assert sess is not None
        countholder = [0] # Counts how many times _collect_grads has been called
        stat = tf.reduce_sum(grads_and_vars[0][1]) # sum of first variable
        def _collect_grads(flat_grad):
            self.comm.Allreduce(flat_grad, buf, op=MPI.SUM)
            np.divide(buf, float(num_tasks), out=buf)
            if countholder[0] % 100 == 0:
                check_synced(sess, self.comm, stat)
            countholder[0] += 1
            return buf

        avg_flat_grad = tf.py_func(_collect_grads, [flat_grad], tf.float32)
        avg_flat_grad.set_shape(flat_grad.shape)
        avg_grads = tf.split(avg_flat_grad, sizes, axis=0)
        avg_grads_and_vars = [(tf.reshape(g, v.shape), v)
                    for g, (_, v) in zip(avg_grads, grads_and_vars)]
        return avg_grads_and_vars

def check_synced(sess, comm, tfstat):
    """
    Check that 'tfstat' evaluates to the same thing on every MPI worker
    """
    localval = sess.run(tfstat)
    vals = comm.gather(localval)
    if comm.rank == 0:
        assert all(val==vals[0] for val in vals[1:])
