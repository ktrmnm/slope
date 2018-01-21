import numpy as np
cimport numpy as np
cimport cython
#import sys

np.import_array()

DOUBLE = np.float64
ctypedef np.float64_t DOUBLE_t


def is_monotone(np.ndarray x, descending=True):
    if descending:
        return np.all(np.diff(x) <= 0)
    else:
        return np.all(np.diff(x) >= 0)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def prox_ordered_L1(np.ndarray[DOUBLE_t, ndim=1] x, np.ndarray[DOUBLE_t, ndim=1] alpha):
    cdef unsigned int n = len(x)
    cdef np.ndarray[DOUBLE_t, ndim=1] z = np.empty(n, dtype=DOUBLE)
    cdef np.ndarray[DOUBLE_t, ndim=1] x_abs = np.abs(x)
    cdef Py_ssize_t[::1] order = np.argsort(-x_abs)

    cdef unsigned int k, j
    cdef unsigned int t = 0
    cdef Py_ssize_t[::1] idx_i = np.empty(n, dtype=np.intp)
    cdef Py_ssize_t[::1] idx_j = np.empty(n, dtype=np.intp)
    cdef np.ndarray[DOUBLE_t, ndim=1] s = np.empty(n, dtype=DOUBLE)
    cdef np.ndarray[DOUBLE_t, ndim=1] w = np.empty(n, dtype=DOUBLE)
    cdef DOUBLE_t w_k

    for k in range(n):
        idx_i[t] = k
        idx_j[t] = k
        s[t] = x_abs[order[k]] - alpha[k]
        w[t] = s[t]

        while (t > 0) and (w[t - 1] <= w[t]):
            t = t - 1
            #idx_i[t] = idx_i[t]
            idx_j[t] = k
            s[t] = s[t] + s[t + 1]
            w[t] = s[t] / (k - idx_i[t] + 1)

        t = t + 1

    for k in range(t):
        w_k = w[k] if w[k] >= 0.0 else 0.0
        for j in range(idx_i[k], idx_j[k] + 1):
            z[order[j]] = w_k if x[order[j]] >= 0.0 else -w_k

    return z


@cython.boundscheck(False)
@cython.wraparound(False)
def _gap(np.ndarray[DOUBLE_t, ndim=1] theta, np.ndarray[DOUBLE_t, ndim=2] X,
        np.ndarray[DOUBLE_t, ndim=1] y, np.ndarray[DOUBLE_t, ndim=1] alpha):
    cdef DOUBLE_t ordered_L1_norm = np.dot(alpha, np.sort(np.abs(theta))[::-1])
    cdef DOUBLE_t gap = np.dot(X.dot(theta), X.dot(theta) - y) + ordered_L1_norm
    return gap


@cython.boundscheck(False)
@cython.wraparound(False)
def _dual_infeasibility(np.ndarray[DOUBLE_t, ndim=1] theta,
                        np.ndarray[DOUBLE_t, ndim=2] X,
                        np.ndarray[DOUBLE_t, ndim=1] y,
                        np.ndarray[DOUBLE_t, ndim=1] alpha,
                        unsigned int p):
    cdef np.ndarray[DOUBLE_t, ndim=1] w = np.dot(X.T, y - X.dot(theta))
    cdef np.ndarray[DOUBLE_t, ndim=1] z = np.sort(np.abs(w))[::-1] - alpha
    cdef unsigned int i
    cdef DOUBLE_t partial = 0, out = 0
    for i in range(p):
        partial += z[i]
        if partial > out:
            out = partial
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _fista_slope(np.ndarray[DOUBLE_t, ndim=2] X,
                np.ndarray[DOUBLE_t, ndim=1] y,
                np.ndarray[DOUBLE_t, ndim=1] alpha,
                learning_rate=None, initial_value=None,
                unsigned int max_iter=1000, DOUBLE_t tol=1e-4, tol_feas=None,
                check_input=False, verbose=False):

    cdef unsigned int n = X.shape[0], p = X.shape[1]
    cdef np.ndarray[DOUBLE_t, ndim=1] alpha_
    cdef np.ndarray[DOUBLE_t, ndim=1] theta
    cdef np.ndarray[DOUBLE_t, ndim=1] theta_new
    cdef np.ndarray[DOUBLE_t, ndim=1] z
    cdef np.ndarray[DOUBLE_t, ndim=1] grad

    # check monotonicity of alpha
    #if not is_monotone(alpha):
    #    print('Warning: alpha is not monotone.')
    if check_input:
        alpha_ = np.sort(alpha)[::-1]
    else:
        alpha_ = alpha

    cdef DOUBLE_t tol_f
    if tol_feas is not None:
        tol_f = <DOUBLE_t> tol_feas
    else:
        tol_g = tol

    # set initial value
    if initial_value is not None:
        theta = <np.ndarray[DOUBLE_t, ndim=1]?> initial_value
        z = <np.ndarray[DOUBLE_t, ndim=1]?> initial_value
    else:
        theta = np.zeros(p, dtype=DOUBLE)
        z = np.zeros(p, dtype=DOUBLE)

    # set learning rate
    cdef DOUBLE_t lr
    cdef DOUBLE_t opnorm
    if learning_rate is not None:
        lr = <DOUBLE_t> learning_rate
    else:
        #opnorm = np.linalg.norm(X, ord=2) ** 2
        opnorm = np.linalg.norm(np.dot(X.T, X), ord=2)
        if opnorm > 0:
            lr = 1.0 / opnorm
        else:
            raise ValueError('Operator norm of X is 0')

    cdef unsigned int iter = 0
    cdef DOUBLE_t gap = tol + 1
    cdef DOUBLE_t feasibility = tol + 1
    cdef DOUBLE_t eta = 0, eta_new, gamma

    while ((gap > tol) or (feasibility > tol_f)) and (iter < max_iter):
        # 1. FISTA update
        grad = np.dot(X.T, X.dot(z) - y)
        theta_new = prox_ordered_L1(z - lr * grad, lr * alpha_)

        eta_new = (1 + np.sqrt(1 + 4 * eta ** 2))/2
        gamma = (1 - eta) / eta_new
        z = (1 - gamma) * theta_new + gamma * theta

        theta = theta_new
        eta = eta_new

        # 2. calculate duality gap & feasibility
        gap = _gap(theta, X, y, alpha_)
        feasibility = _dual_infeasibility(theta, X, y, alpha_, p)

        #if verbose:
        #    print('iter {0}\t gap: {1:.4f}, infeasibility: {2:.4f}'.format(iter, gap, feasibility))

        iter += 1

    if verbose and iter == max_iter:
        print(
            'Warning: FISTA did not converge at #iter={0}. (Gap: {1}, Infeasibility: {2})'
            .format(iter, gap, feasibility)
            )

    return theta, gap, iter
