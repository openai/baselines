from mpi4py import MPI
import baselines.common.tf_util as tf_utils
import tensorflow as tf
import numpy as np


class MpiAdam(object):
    def __init__(self, var_list, *, beta1=0.9, beta2=0.999, epsilon=1e-08, scale_grad_by_procs=True, comm=None,
                 sess=None):
        """
        A parallel MPI implementation of the Adam optimizer for TensorFlow
        https://arxiv.org/abs/1412.6980

        :param var_list: ([TensorFlow Tensor]) the variables
        :param beta1: (float) Adam beta1 parameter
        :param beta2: (float) Adam beta1 parameter
        :param epsilon: (float) to help with preventing arithmetic issues
        :param scale_grad_by_procs: (bool) if the scaling should be done by processes
        :param comm: (MPI Communicators) if None, MPI.COMM_WORLD
        :param sess: (TensorFlow Session) if None, tf.get_default_session()
        """
        self.var_list = var_list
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.scale_grad_by_procs = scale_grad_by_procs
        size = sum(tf_utils.numel(v) for v in var_list)
        self.m = np.zeros(size, 'float32')
        self.v = np.zeros(size, 'float32')
        self.t = 0
        self.setfromflat = tf_utils.SetFromFlat(var_list, sess=sess)
        self.getflat = tf_utils.GetFlat(var_list, sess=sess)
        self.comm = MPI.COMM_WORLD if comm is None else comm

    def update(self, localg, stepsize):
        """
        update the values of the graph

        :param localg: (numpy float) the gradiant
        :param stepsize: (float) the stepsize for the update
        """
        if self.t % 100 == 0:
            self.check_synced()
        localg = localg.astype('float32')
        globalg = np.zeros_like(localg)
        self.comm.Allreduce(localg, globalg, op=MPI.SUM)
        if self.scale_grad_by_procs:
            globalg /= self.comm.Get_size()

        self.t += 1
        a = stepsize * np.sqrt(1 - self.beta2**self.t)/(1 - self.beta1**self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = (- a) * self.m / (np.sqrt(self.v) + self.epsilon)
        self.setfromflat(self.getflat() + step)

    def sync(self):
        """
        syncronize the MPI threads
        """
        theta = self.getflat()
        self.comm.Bcast(theta, root=0)
        self.setfromflat(theta)

    def check_synced(self):
        """
        confirm the MPI threads are synced
        """
        if self.comm.Get_rank() == 0:  # this is root
            theta = self.getflat()
            self.comm.Bcast(theta, root=0)
        else:
            thetalocal = self.getflat()
            thetaroot = np.empty_like(thetalocal)
            self.comm.Bcast(thetaroot, root=0)
            assert (thetaroot == thetalocal).all(), (thetaroot, thetalocal)


@tf_utils.in_session
def test_mpi_adam():
    """
    tests the MpiAdam object's functionality
    """
    np.random.seed(0)
    tf.set_random_seed(0)

    a = tf.Variable(np.random.randn(3).astype('float32'))
    b = tf.Variable(np.random.randn(2, 5).astype('float32'))
    loss = tf.reduce_sum(tf.square(a)) + tf.reduce_sum(tf.sin(b))

    stepsize = 1e-2
    update_op = tf.train.AdamOptimizer(stepsize).minimize(loss)
    do_update = tf_utils.function([], loss, updates=[update_op])

    tf.get_default_session().run(tf.global_variables_initializer())
    for i in range(10):
        print(i, do_update())

    tf.set_random_seed(0)
    tf.get_default_session().run(tf.global_variables_initializer())

    var_list = [a, b]
    lossandgrad = tf_utils.function([], [loss, tf_utils.flatgrad(loss, var_list)], updates=[update_op])
    adam = MpiAdam(var_list)

    for i in range(10):
        l, g = lossandgrad()
        adam.update(g, stepsize)
        print(i, l)
