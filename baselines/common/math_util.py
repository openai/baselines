import numpy as np
import scipy.signal


def discount(vector, gamma):
    """
    computes discounted sums along 0th dimension of vector x.
        y[t] = x[t] + gamma*x[t+1] + gamma^2*x[t+2] + ... + gamma^k x[t+k],
                where k = len(x) - t - 1

    :param vector: (numpy array) the input vector
    :param gamma: (float) the discount value
    :return: (numpy Number) the output vector
    """
    assert vector.ndim >= 1
    return scipy.signal.lfilter([1], [1, -gamma], vector[::-1], axis=0)[::-1]


def explained_variance(y_pred, y_true):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    :param y_pred: (numpy Number) the prediction
    :param y_true: (numpy Number) the expected value
    :return: (float) explained variance of ypred and y
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


def explained_variance_2d(ypred, y):
    """
    Computes fraction of variance that ypred explains about y, for 2D arrays.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    :param ypred: (numpy Number) the prediction
    :param y: (numpy Number) the expected value
    :return: (float) explained variance of ypred and y
    """
    assert y.ndim == 2 and ypred.ndim == 2
    vary = np.var(y, axis=0)
    out = 1 - np.var(y - ypred) / vary
    out[vary < 1e-10] = 0
    return out


def flatten_arrays(arrs):
    """
    flattens a list of arrays down to 1D

    :param arrs: ([numpy Number]) arrays
    :return: (numpy Number) 1D flattend array
    """
    return np.concatenate([arr.flat for arr in arrs])


def unflatten_vector(vec, shapes):
    """
    reshape a flattened array

    :param vec: (numpy Number) 1D arrays
    :param shapes: (tuple)
    :return: ([numpy Number]) reshaped array
    """
    i = 0
    arrs = []
    for shape in shapes:
        size = np.prod(shape)
        arr = vec[i:i + size].reshape(shape)
        arrs.append(arr)
        i += size
    return arrs


def discount_with_boundaries(x, new, gamma):
    """
    computes discounted sums along 0th dimension of x, while taking into account the start of each episode.
        y[t] = x[t] + gamma*x[t+1] + gamma^2*x[t+2] + ... + gamma^k x[t+k],
                where k = len(x) - t - 1

    :param x: (numpy Number) the input vector
    :param new: (numpy Number) 2d array of bools, indicating when a new episode has started
    :param gamma: (float) the discount value
    :return: (numpy Number) the output vector
    """
    y = np.zeros_like(x)
    n_samples = x.shape[0]
    y[n_samples - 1] = x[n_samples - 1]
    for step in range(n_samples - 2, -1, -1):
        y[step] = x[step] + gamma * y[step + 1] * (1 - new[step + 1])
    return y


def test_discount_with_boundaries():
    """
    test the discount_with_boundaries function
    """
    gamma = 0.9
    x = np.array([1.0, 2.0, 3.0, 4.0], 'float32')
    starts = [1.0, 0.0, 0.0, 1.0]
    y = discount_with_boundaries(x, starts, gamma)
    assert np.allclose(y, [1 + gamma * 2 + gamma ** 2 * 3, 2 + gamma * 3, 3, 4])
