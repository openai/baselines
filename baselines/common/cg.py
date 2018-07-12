import numpy as np


def conjugate_gradient(f_ax, b, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10):
    """
    conjugate gradient calculation (Ax = b), bases on
    https://epubs.siam.org/doi/book/10.1137/1.9781611971446 Demmel p 312

    :param f_ax: (function) The function describing the Matrix A dot the vector x
                 (x being the input parameter of the function)
    :param b: (numpy float) vector b, where Ax = b
    :param cg_iters: (int) the maximum number of iterations for converging
    :param callback: (function) callback the values of x while converging
    :param verbose: (bool) print extra information
    :param residual_tol: (float) the break point if the residual is below this value
    :return: (numpy float) vector x, where Ax = b
    """
    p = b.copy()  # the first basis vector
    r = b.copy()  # the residual
    x = np.zeros_like(b)  # vector x, where Ax = b
    rdotr = r.dot(r)  # L2 norm of the residual

    fmtstr = "%10i %10.3g %10.3g"
    titlestr = "%10s %10s %10s"
    if verbose:
        print(titlestr % ("iter", "residual norm", "soln norm"))

    for i in range(cg_iters):
        if callback is not None:
            callback(x)
        if verbose:
            print(fmtstr % (i, rdotr, np.linalg.norm(x)))
        z = f_ax(p)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break

    if callback is not None:
        callback(x)
    if verbose:
        print(fmtstr % (i + 1, rdotr, np.linalg.norm(x)))
    return x
