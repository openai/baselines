import numpy as np
import scipy.signal


def discount(x, gamma):
    """
    computes discounted sums along 0th dimension of x.

    inputs
    ------
    x: ndarray
    gamma: float

    outputs
    -------
    y: ndarray with same shape as x, satisfying

        y[t] = x[t] + gamma*x[t+1] + gamma^2*x[t+2] + ... + gamma^k x[t+k],
                where k = len(x) - t - 1

    """
    assert x.ndim >= 1
    return scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]

def explained_variance(ypred,y):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary

def explained_variance_2d(ypred, y):
    assert y.ndim == 2 and ypred.ndim == 2
    vary = np.var(y, axis=0)
    out = 1 - np.var(y-ypred)/vary
    out[vary < 1e-10] = 0
    return out

def ncc(ypred, y):
    return np.corrcoef(ypred, y)[1,0]

def flatten_arrays(arrs):
    return np.concatenate([arr.flat for arr in arrs])

def unflatten_vector(vec, shapes):
    i=0
    arrs = []
    for shape in shapes:
        size = np.prod(shape)
        arr = vec[i:i+size].reshape(shape)
        arrs.append(arr)
        i += size
    return arrs

def discount_with_boundaries(X, New, gamma):
    """
    X: 2d array of floats, time x features
    New: 2d array of bools, indicating when a new episode has started
    """
    Y = np.zeros_like(X)
    T = X.shape[0]
    Y[T-1] = X[T-1]
    for t in range(T-2, -1, -1):
        Y[t] = X[t] + gamma * Y[t+1] * (1 - New[t+1])
    return Y

def test_discount_with_boundaries():
    gamma=0.9
    x = np.array([1.0, 2.0, 3.0, 4.0], 'float32')
    starts = [1.0, 0.0, 0.0, 1.0]
    y = discount_with_boundaries(x, starts, gamma)
    assert np.allclose(y, [
        1 + gamma * 2 + gamma**2 * 3,
        2 + gamma * 3,
        3,
        4
    ])