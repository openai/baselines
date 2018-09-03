import numpy as np
import scipy.signal


def discount(vector, gamma):
    """
    computes discounted sums along 0th dimension of vector x.
        y[t] = x[t] + gamma*x[t+1] + gamma^2*x[t+2] + ... + gamma^k x[t+k],
                where k = len(x) - t - 1

    :param vector: (np.ndarray) the input vector
    :param gamma: (float) the discount value
    :return: (np.ndarray) the output vector
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

    :param y_pred: (np.ndarray) the prediction
    :param y_true: (np.ndarray) the expected value
    :return: (float) explained variance of ypred and y
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


def explained_variance_2d(y_pred, y_true):
    """
    Computes fraction of variance that ypred explains about y, for 2D arrays.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    :param y_pred: (np.ndarray) the prediction
    :param y_true: (np.ndarray) the expected value
    :return: (float) explained variance of ypred and y
    """
    assert y_true.ndim == 2 and y_pred.ndim == 2
    var_y = np.var(y_true, axis=0)
    explained_var = 1 - np.var(y_true - y_pred) / var_y
    explained_var[var_y < 1e-10] = 0
    return explained_var


def flatten_arrays(arrs):
    """
    flattens a list of arrays down to 1D

    :param arrs: ([np.ndarray]) arrays
    :return: (np.ndarray) 1D flattend array
    """
    return np.concatenate([arr.flat for arr in arrs])


def unflatten_vector(vec, shapes):
    """
    reshape a flattened array

    :param vec: (np.ndarray) 1D arrays
    :param shapes: (tuple)
    :return: ([np.ndarray]) reshaped array
    """
    i = 0
    arrs = []
    for shape in shapes:
        size = np.prod(shape)
        arr = vec[i:i + size].reshape(shape)
        arrs.append(arr)
        i += size
    return arrs


def discount_with_boundaries(rewards, episode_starts, gamma):
    """
    computes discounted sums along 0th dimension of x (reward), while taking into account the start of each episode.
        y[t] = x[t] + gamma*x[t+1] + gamma^2*x[t+2] + ... + gamma^k x[t+k],
                where k = len(x) - t - 1

    :param rewards: (np.ndarray) the input vector (rewards)
    :param episode_starts: (np.ndarray) 2d array of bools, indicating when a new episode has started
    :param gamma: (float) the discount factor
    :return: (np.ndarray) the output vector (discounted rewards)
    """
    discounted_rewards = np.zeros_like(rewards)
    n_samples = rewards.shape[0]
    discounted_rewards[n_samples - 1] = rewards[n_samples - 1]
    for step in range(n_samples - 2, -1, -1):
        discounted_rewards[step] = rewards[step] + gamma * discounted_rewards[step + 1] * (1 - episode_starts[step + 1])
    return discounted_rewards
