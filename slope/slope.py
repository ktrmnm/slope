import warnings

import numpy as np
from scipy import sparse
from scipy.stats import norm

from .base import LinearModel, _preprocess_data, _pre_fit
from sklearn.base import RegressorMixin
from sklearn.utils import check_array, check_X_y
from sklearn.utils.extmath import safe_sparse_dot

from ._prox import is_monotone, _fista_slope


def normal_decay(n_features, q=0.05):
    """ Weight vector for the ordered L1 norm by normal percentile

    Parameters
    ----------
    n_features : int
        Number of features

    q : float

    Returns
    -------
    alpha_decay : array, shape (n_features, )
    """

    if n_features <= 0:
        raise ValueError('n_features must be set >= 1')
    if (q <= 0) or (q >= 1):
        raise ValueError('significance level q must be chosen from (0, 1)')

    grid = 1.0 - (q / (2 * n_features)) * np.arange(1, n_features + 1)

    return norm.ppf(grid)


def _slope_alpha_grid(X, y, Xy=None, alpha_decay=None, fit_intercept=True, eps=1e-3,
                n_alphas=100, normalize=False, copy_X=True, return_alpha_decay=False):
    """ Comupute the grid of alpha values for Slope parameter search.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.

    y : ndarray, shape (n_samples,)
        Target values.

    alpha_dacay : ndarray, shape (n_features)
        Decay rate for the ordered L1 norm. Given alpha, the parameter of the
        ordered L1 norm is calculated as alpha * alpha_decay.
        If None alpha_decay is set to normal_decay(n_features).

    For more detailed explanations of the other parameters, see the corresponding
    function in skearn.linear_model.

    Returns
    -------
    alpha_grid : ndarray, shape (n_alphas)

    alpha_decay_ : ndarray, shape (n_features)
    """

    n_samples = len(y)

    sparse_center = False
    if Xy is None:
        X_sparse = sparse.isspmatrix(X)
        sparse_center = X_sparse and (fit_intercept or normalize)
        X = check_array(X, 'csc',
                        copy=(copy_X and fit_intercept and not X_sparse))
        if not X_sparse:
            # X can be touched inplace thanks to the above line
            X, y, _, _, _ = _preprocess_data(X, y, fit_intercept,
                                             normalize, copy=False)
        Xy = safe_sparse_dot(X.T, y, dense_output=True)

        if sparse_center:
            # Workaround to find alpha_max for sparse matrices.
            # since we should not destroy the sparsity of such matrices.
            _, _, X_offset, _, X_scale = _preprocess_data(X, y, fit_intercept,
                                                      normalize,
                                                      return_mean=True)
            mean_dot = X_offset * np.sum(y)

    if Xy.ndim == 1:
        Xy = Xy[:, np.newaxis]

    if sparse_center:
        if fit_intercept:
            Xy -= mean_dot[:, np.newaxis]
        if normalize:
            Xy /= X_scale[:, np.newaxis]

    if alpha_decay is None:
        alpha_decay_ = normal_decay(len(Xy))
    else:
        alpha_decay_ = alpha_decay

    #u = np.sum(np.flip(np.sort(np.abs(Xy), axis=0), axis=0), axis=1)
    u = np.sort(np.abs(Xy) / n_samples)[::-1]
    alpha_max = np.max(np.cumsum(u) / np.cumsum(alpha_decay_))

    if alpha_max <= np.finfo(float).resolution:
        alphas = np.empty(n_alphas)
        alphas.fill(np.finfo(float).resolution)
        return alphas

    alphas = np.logspace(np.log10(alpha_max * eps), np.log10(alpha_max), num=n_alphas)[::-1]
    if return_alpha_decay:
        return alphas, alpha_decay_
    else:
        return alphas


def slope_path(X, y, eps=1e-3, n_alphas=100, alphas=None, alpha_decay=None,
        check_input=False, copy_X=True, coef_init=None, verbose=False,
        return_n_iter=False, **params):
    """Compute Slope path with accelerated proximal gradient method (FISTA)

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data. Currently, sparse matrix is not supported.

    y : array-like, shape (n_samples,)
        Target values.

    eps : float

    n_alphas : int

    alpha_decay : ndarray, shape (n_features)
        Decay rate for the ordered L1 norm. Given alpha, the parameter of the
        ordered L1 norm is calculated as alpha * alpha_decay.
        If None alpha_decay is set to normal_decay(n_features).

    check_input : boolean

    copy_X : boolean, optional, default True

    coef_init : array, shape (n_features, ) | None
        The initial values of the coefficients.

    verbose : boolean

    return_n_iter : boolean

    Returns
    -------
    alphas : array, shape (n_alphas,)
        The alphas along the path where models are computed.
    coefs : array, shape (n_features, n_alphas) or (n_outputs, n_features, n_alphas)
        Coefficients along the path.
    dual_gaps : array, shape (n_alphas,)
        The dual gaps at the end of the optimization for each alpha.
    n_iters : array-like, shape (n_alphas,)
        The number of iterations taken by the coordinate descent optimizer to
        reach the specified tolerance for each alpha.
    alpha_decay_ : array, shape (n_features,)

    """
    if check_input:
        X = check_array(X, accept_sparse=False, dtype=np.float64, copy=copy_X)
        y = check_array(y, accept_sparse=False, dtype=X.dtype.type, copy=False,
                        ensure_2d=False)

    n_samples, n_features = X.shape

    if check_input:
        X, y, X_offset, y_offset, X_scale, _, _ = \
            _pre_fit(X, y, None, False, normalize=False, fit_intercept=False, copy=False)

    if alpha_decay is None:
        alpha_decay_ = normal_decay(n_features)
    else:
        if (len(alpha_decay) != n_features) and not is_monotone(alpha_decay, decending=True):
            raise ValueError(
                'alpha_decay must be monotone non-increasing vector of size n_features.')
        alpha_decay_ = alpha_decay
    #self.alpha_decay = alpha_decay_

    if alphas is None:
        alphas = _slope_alpha_grid(X, y, alpha_decay=alpha_decay_,
                fit_intercept=False, eps=eps, n_alphas=n_alphas,
                normalize=False, copy_X=False)
    else:
        alphas = np.sort(alphas)[::-1]

    n_alphas = len(alphas)
    tol = params.get('tol', 1e-4)
    max_iter = params.get('max_iter', 1000)
    dual_gaps = np.empty(n_alphas)
    n_iters = []
    lr = 1.0 / np.linalg.norm(np.dot(X.T, X), ord=2)

    coefs = np.empty((n_features, n_alphas), dtype=X.dtype)
    if coef_init is None:
        coef_ = np.zeros(coefs.shape[:-1], dtype=X.dtype)
    else:
        coef_ = coef_init

    for i, alpha in enumerate(alphas):
        reg_param = alpha * alpha_decay_ * n_samples

        coef_, gap_, n_iter_ = _fista_slope(X, y, reg_param, max_iter=max_iter,
            initial_value=coef_, learning_rate = lr, tol=tol, verbose=verbose)

        coefs[:, i] = coef_
        dual_gaps[i] = gap_
        n_iters.append(n_iter_)

    if return_n_iter:
        return alphas, coefs, dual_gaps, n_iters
    else:
        return alphas, coefs, dual_gaps


class Slope(LinearModel, RegressorMixin):
    """ Slope [1] for linear regression

    [1] M. Bogdan, E. van den Berg, C. Sabatti, W. Su, and E. J. Candès.
    SLOPE—Adaptive variable selection via convex optimization.
    The Annals of Applied Statistics, 9(3):1103--1140, 2015.

    """
    path = staticmethod(slope_path)

    def __init__(self, alpha=1.0, alpha_decay=None, fit_intercept=True,
                normalize=False, max_iter=1000, copy_X=True, tol=1e-4,
                warm_start=False):
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        #self.precompute = precompute
        self.max_iter = max_iter
        self.copy_X = copy_X
        self.tol = tol
        self.warm_start = warm_start

    def fit(self, X, y, check_input=True):
        if self.alpha == 0:
            warnings.warn("With alpha=0, this algorithm does not converge "
                          "well. You are advised to use the LinearRegression "
                          "estimator", stacklevel=2)

        if isinstance(self.precompute, six.string_types):
            raise ValueError('precompute should be one of True, False or'
                             ' array-like. Got %r' % self.precompute)

        if check_input:
            X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64,
                             copy=self.copy_X and self.fit_intercept)
            y = check_array(y, copy=False, dtype=X.dtype.type, ensure_2d=False)

        X, y, X_offset, y_offset, X_scale, precompute, Xy = \
            _pre_fit(X, y, None, False, self.normalize, self.fit_intercept, copy=False)

        if y.ndim == 1:
            y = y[:, np.newaxis]
        if Xy is not None and Xy.ndim == 1:
            Xy = Xy[:, np.newaxis]

        n_samples, n_features = X.shape

        if not self.warm_start or not hasattr(self, "coef_"):
            coef_ = np.zeros(n_deatures, dtype=X.dtype)
        else:
            coef_ = self.coef_
            if coef_.ndim == 1:
                coef_ = coef_[np.newaxis, :]

        if self.alpha_decay is None:
            self.alpha_decay = normal_decay(n_features)

        #dual_gaps_ = np.zeros(n_targets, dtype=X.dtype)
        self.n_iter_ = []

        if Xy is not None:
            Xy_ = Xy
        else:
            Xy_ = None
            _, coef_, dual_gap, n_iter = \
            self.path(X, y[:, k],
                    n_alphas=None, alphas=[self.alpha], alpha_decay=self.alpha_decay,
                    copy_X=True, coef_init=coef_, verbose=False, return_n_iter=True,
                    tol=self.tol, max_iter=self.max_iter)
        self.coef_ = coef_
        self.dual_gap_ = dual_gap
        self.n_iter_ = n_iter

        self._set_intercept(X_offset, y_offset, X_scale)

        # workaround since _set_intercept will cast self.coef_ into X.dtype
        self.coef_ = np.asarray(self.coef_, dtype=X.dtype)

        return self
