"""HAR-RV (Corsi 2009) and the naive-persistence baseline.

HAR-RV regresses next-day realized variance on trailing daily, weekly, and
monthly realized-variance averages by closed-form OLS. Naive persistence
just carries yesterday's realized variance forward as tomorrow's forecast.
"""

import numpy as np


def fit_ols(X, y):
    """Closed-form OLS via the normal equations: beta = (X'X)^-1 X'y."""
    return np.linalg.solve(X.T @ X, X.T @ y)


def har_features(dataset, mask):
    """Builds the [intercept, daily_rv, weekly_rv, monthly_rv] design matrix
    for the rows selected by `mask`.
    """
    daily = dataset["daily_rv"][mask]
    weekly = dataset["weekly_rv"][mask]
    monthly = dataset["monthly_rv"][mask]
    intercept = np.ones_like(daily)
    return np.column_stack([intercept, daily, weekly, monthly])


def fit_har(dataset, train_mask):
    """Fits HAR-RV on the training rows, returning the OLS coefficients."""
    X_train = har_features(dataset, train_mask)
    y_train = dataset["rv"][train_mask]
    return fit_ols(X_train, y_train)


def predict_har(dataset, mask, beta):
    """Predicts realized variance for the rows selected by `mask`."""
    return har_features(dataset, mask) @ beta


def naive_forecast(dataset, mask):
    """Naive persistence: the forecast for day t is rv at t - 1.

    This is exactly `daily_rv`, since data.build_dataset already defines
    daily_rv[t] = rv[t - 1].
    """
    return dataset["daily_rv"][mask]
