import numpy as np


def conjugate_gradient(f_ax, b_vec, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10):
    """
    conjugate gradient calculation (Ax = b), bases on
    https://epubs.siam.org/doi/book/10.1137/1.9781611971446 Demmel p 312

    :param f_ax: (function) The function describing the Matrix A dot the vector x
                 (x being the input parameter of the function)
    :param b_vec: (numpy float) vector b, where Ax = b
    :param cg_iters: (int) the maximum number of iterations for converging
    :param callback: (function) callback the values of x while converging
    :param verbose: (bool) print extra information
    :param residual_tol: (float) the break point if the residual is below this value
    :return: (numpy float) vector x, where Ax = b
    """
    first_basis_vect = b_vec.copy()  # the first basis vector
    residual = b_vec.copy()  # the residual
    x_var = np.zeros_like(b_vec)  # vector x, where Ax = b
    residual_dot_residual = residual.dot(residual)  # L2 norm of the residual

    fmt_str = "%10i %10.3g %10.3g"
    title_str = "%10s %10s %10s"
    if verbose:
        print(title_str % ("iter", "residual norm", "soln norm"))

    for i in range(cg_iters):
        if callback is not None:
            callback(x_var)
        if verbose:
            print(fmt_str % (i, residual_dot_residual, np.linalg.norm(x_var)))
        z_var = f_ax(first_basis_vect)
        v_var = residual_dot_residual / first_basis_vect.dot(z_var)
        x_var += v_var * first_basis_vect
        residual -= v_var * z_var
        new_residual_dot_residual = residual.dot(residual)
        mu_val = new_residual_dot_residual / residual_dot_residual
        first_basis_vect = residual + mu_val * first_basis_vect

        residual_dot_residual = new_residual_dot_residual
        if residual_dot_residual < residual_tol:
            break

    if callback is not None:
        callback(x_var)
    if verbose:
        print(fmt_str % (i + 1, residual_dot_residual, np.linalg.norm(x_var)))
    return x_var
