"""GARCH(1,1) fit by Gaussian quasi-maximum likelihood.

sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2, fit on a
training window of returns and then rolled forward one step at a time
through a test window using only returns realized up to each point.
"""

import numpy as np
from scipy.optimize import minimize

MAX_PERSISTENCE = 0.999


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _params_from_unconstrained(theta):
    """Maps unconstrained reals to (omega, alpha, beta).

    omega = exp(log_omega) > 0. alpha and beta are built from sigmoids so
    that alpha >= 0, beta >= 0, and alpha + beta <= MAX_PERSISTENCE < 1
    hold for every theta, with no bounded optimizer needed.
    """
    log_omega, x, y = theta
    omega = np.exp(log_omega)
    alpha = _sigmoid(x) * MAX_PERSISTENCE
    beta = _sigmoid(y) * (MAX_PERSISTENCE - alpha)
    return omega, alpha, beta


def _sigma2_path(omega, alpha, beta, r):
    n = len(r)
    sigma2 = np.empty(n)
    sigma2[0] = np.var(r)
    for t in range(1, n):
        sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
    return sigma2


def _neg_log_likelihood(theta, r):
    omega, alpha, beta = _params_from_unconstrained(theta)
    sigma2 = np.maximum(_sigma2_path(omega, alpha, beta, r), 1e-12)
    log_lik = -0.5 * (np.log(sigma2) + r ** 2 / sigma2)
    return -np.sum(log_lik)


def fit_garch(r):
    """Fits GARCH(1,1) on the return series `r` by QML via L-BFGS-B.

    Returns (omega, alpha, beta, optimize_result).
    """
    theta0 = np.array([np.log(np.var(r) * 0.05), 0.0, 0.0])
    result = minimize(_neg_log_likelihood, theta0, args=(r,), method="L-BFGS-B")
    omega, alpha, beta = _params_from_unconstrained(result.x)
    return omega, alpha, beta, result


def rolling_forecast(omega, alpha, beta, r_train, r_test):
    """Rolls the GARCH(1,1) recursion forward through the test period.

    Returns one one-step-ahead variance forecast per test point. The first
    test forecast is seeded from the end of the training recursion; every
    later forecast uses only returns realized in the test window up to
    that point, never a later one.
    """
    sigma2_train = _sigma2_path(omega, alpha, beta, r_train)

    n_test = len(r_test)
    forecasts = np.empty(n_test)
    prev_r2 = r_train[-1] ** 2
    prev_sigma2 = sigma2_train[-1]
    for t in range(n_test):
        forecasts[t] = omega + alpha * prev_r2 + beta * prev_sigma2
        prev_r2 = r_test[t] ** 2
        prev_sigma2 = forecasts[t]
    return forecasts
