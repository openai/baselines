import numpy as np
import tensorflow as tf
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

class MpiAdamOptimizer(tf.Module):
    """Adam optimizer that averages gradients across mpi processes."""
    def __init__(self, comm, var_list):
        self.var_list = var_list
        self.comm = comm
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-08
        self.t = tf.Variable(0, name='step', dtype=tf.int32)
        var_shapes = [v.shape.as_list() for v in var_list]
        self.var_sizes = [int(np.prod(s)) for s in var_shapes]
        self.flat_var_size = sum(self.var_sizes)
        self.m = tf.Variable(np.zeros(self.flat_var_size, 'float32'))
        self.v = tf.Variable(np.zeros(self.flat_var_size, 'float32'))

    def apply_gradients(self, flat_grad, lr):
        buf = np.zeros(self.flat_var_size, np.float32)
        self.comm.Allreduce(flat_grad.numpy(), buf, op=MPI.SUM)
        avg_flat_grad = np.divide(buf, float(self.comm.Get_size()))
        self._apply_gradients(tf.constant(avg_flat_grad), lr)
        if self.t.numpy() % 100 == 0:
            check_synced(tf.reduce_sum(self.var_list[0]).numpy())

    @tf.function
    def _apply_gradients(self, avg_flat_grad, lr):
        self.t.assign_add(1)
        t = tf.cast(self.t, tf.float32)
        a = lr * tf.math.sqrt(1 - tf.math.pow(self.beta2, t)) / (1 - tf.math.pow(self.beta1, t))
        self.m.assign(self.beta1 * self.m + (1 - self.beta1) * avg_flat_grad)
        self.v.assign(self.beta2 * self.v + (1 - self.beta2) * tf.math.square(avg_flat_grad))
        flat_step = (- a) * self.m / (tf.math.sqrt(self.v) + self.epsilon)
        var_steps = tf.split(flat_step, self.var_sizes, axis=0)
        for var_step, var in zip(var_steps, self.var_list):
            var.assign_add(tf.reshape(var_step, var.shape))


def check_synced(localval, comm=None):
    """
    It's common to forget to initialize your variables to the same values, or
    (less commonly) if you update them in some other way than adam, to get them out of sync.
    This function checks that variables on all MPI workers are the same, and raises
    an AssertionError otherwise

    Arguments:
        comm: MPI communicator
        localval: list of local variables (list of variables on current worker to be compared with the other workers)
    """
    comm = comm or MPI.COMM_WORLD
    vals = comm.gather(localval)
    if comm.rank == 0:
        assert all(val==vals[0] for val in vals[1:]),\
            'MpiAdamOptimizer detected that different workers have different weights: {}'.format(vals)
