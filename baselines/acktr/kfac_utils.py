import tensorflow as tf
import numpy as np


def gmatmul(a, b, transpose_a=False, transpose_b=False, reduce_dim=None):
    if reduce_dim == None:
        # general batch matmul
        if len(a.get_shape()) == 3 and len(b.get_shape()) == 3:
            return tf.batch_matmul(a, b, adj_x=transpose_a, adj_y=transpose_b)
        elif len(a.get_shape()) == 3 and len(b.get_shape()) == 2:
            if transpose_b:
                N = b.get_shape()[0].value
            else:
                N = b.get_shape()[1].value
            B = a.get_shape()[0].value
            if transpose_a:
                K = a.get_shape()[1].value
                a = tf.reshape(tf.transpose(a, [0, 2, 1]), [-1, K])
            else:
                K = a.get_shape()[-1].value
                a = tf.reshape(a, [-1, K])
            result = tf.matmul(a, b, transpose_b=transpose_b)
            result = tf.reshape(result, [B, -1, N])
            return result
        elif len(a.get_shape()) == 2 and len(b.get_shape()) == 3:
            if transpose_a:
                M = a.get_shape()[1].value
            else:
                M = a.get_shape()[0].value
            B = b.get_shape()[0].value
            if transpose_b:
                K = b.get_shape()[-1].value
                b = tf.transpose(tf.reshape(b, [-1, K]), [1, 0])
            else:
                K = b.get_shape()[1].value
                b = tf.transpose(tf.reshape(
                    tf.transpose(b, [0, 2, 1]), [-1, K]), [1, 0])
            result = tf.matmul(a, b, transpose_a=transpose_a)
            result = tf.transpose(tf.reshape(result, [M, B, -1]), [1, 0, 2])
            return result
        else:
            return tf.matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b)
    else:
        # weird batch matmul
        if len(a.get_shape()) == 2 and len(b.get_shape()) > 2:
            # reshape reduce_dim to the left most dim in b
            b_shape = b.get_shape()
            if reduce_dim != 0:
                b_dims = list(range(len(b_shape)))
                b_dims.remove(reduce_dim)
                b_dims.insert(0, reduce_dim)
                b = tf.transpose(b, b_dims)
            b_t_shape = b.get_shape()
            b = tf.reshape(b, [int(b_shape[reduce_dim]), -1])
            result = tf.matmul(a, b, transpose_a=transpose_a,
                               transpose_b=transpose_b)
            result = tf.reshape(result, b_t_shape)
            if reduce_dim != 0:
                b_dims = list(range(len(b_shape)))
                b_dims.remove(0)
                b_dims.insert(reduce_dim, 0)
                result = tf.transpose(result, b_dims)
            return result

        elif len(a.get_shape()) > 2 and len(b.get_shape()) == 2:
            # reshape reduce_dim to the right most dim in a
            a_shape = a.get_shape()
            outter_dim = len(a_shape) - 1
            reduce_dim = len(a_shape) - reduce_dim - 1
            if reduce_dim != outter_dim:
                a_dims = list(range(len(a_shape)))
                a_dims.remove(reduce_dim)
                a_dims.insert(outter_dim, reduce_dim)
                a = tf.transpose(a, a_dims)
            a_t_shape = a.get_shape()
            a = tf.reshape(a, [-1, int(a_shape[reduce_dim])])
            result = tf.matmul(a, b, transpose_a=transpose_a,
                               transpose_b=transpose_b)
            result = tf.reshape(result, a_t_shape)
            if reduce_dim != outter_dim:
                a_dims = list(range(len(a_shape)))
                a_dims.remove(outter_dim)
                a_dims.insert(reduce_dim, outter_dim)
                result = tf.transpose(result, a_dims)
            return result

        elif len(a.get_shape()) == 2 and len(b.get_shape()) == 2:
            return tf.matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b)

        assert False, 'something went wrong'


def clipoutNeg(vec, threshold=1e-6):
    mask = tf.cast(vec > threshold, tf.float32)
    return mask * vec


def detectMinVal(input_mat, var, threshold=1e-6, name='', debug=False):
    eigen_min = tf.reduce_min(input_mat)
    eigen_max = tf.reduce_max(input_mat)
    eigen_ratio = eigen_max / eigen_min
    input_mat_clipped = clipoutNeg(input_mat, threshold)

    if debug:
        input_mat_clipped = tf.cond(tf.logical_or(tf.greater(eigen_ratio, 0.), tf.less(eigen_ratio, -500)), lambda: input_mat_clipped, lambda: tf.Print(
            input_mat_clipped, [tf.convert_to_tensor('screwed ratio ' + name + ' eigen values!!!'), tf.convert_to_tensor(var.name), eigen_min, eigen_max, eigen_ratio]))

    return input_mat_clipped


def factorReshape(Q, e, grad, facIndx=0, ftype='act'):
    grad_shape = grad.get_shape()
    if ftype == 'act':
        assert e.get_shape()[0] == grad_shape[facIndx]
        expanded_shape = [1, ] * len(grad_shape)
        expanded_shape[facIndx] = -1
        e = tf.reshape(e, expanded_shape)
    if ftype == 'grad':
        assert e.get_shape()[0] == grad_shape[len(grad_shape) - facIndx - 1]
        expanded_shape = [1, ] * len(grad_shape)
        expanded_shape[len(grad_shape) - facIndx - 1] = -1
        e = tf.reshape(e, expanded_shape)

    return Q, e
