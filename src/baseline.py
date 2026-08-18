"""Runs the GARCH, HAR-RV, and naive-persistence baselines on the fixed
out-of-sample split and reports QLIKE and MSE for each.
"""

import numpy as np

from data import build_dataset
from garch import fit_garch, rolling_forecast
from har import fit_har, naive_forecast, predict_har
from split import SPLIT_DATE, split_dataset


def qlike(actual, pred, pred_floor):
    """QLIKE loss: actual/pred - log(actual/pred) - 1, averaged.

    Scale-free and penalizes under-prediction of variance more than
    over-prediction, which is why it is the headline metric here rather
    than MSE. Undefined when actual is exactly zero (log(0)), which the
    squared-daily-return proxy used here occasionally is on days the
    market return happened to round to zero; those rows are excluded from
    the QLIKE average (not from MSE, which handles zero fine) and the
    excluded count is reported alongside the number. Predictions of
    exactly zero (naive persistence, after a zero-return day) are clipped
    to `pred_floor` rather than left to divide by zero.
    """
    pred = np.maximum(pred, pred_floor)
    nonzero = actual > 0
    n_excluded = int(np.size(actual) - np.sum(nonzero))
    ratio = actual[nonzero] / pred[nonzero]
    value = float(np.mean(ratio - np.log(ratio) - 1.0))
    return value, n_excluded


def mse(actual, pred):
    return float(np.mean((actual - pred) ** 2))


def report(name, actual, pred, pred_floor):
    n_clipped = int(np.sum(pred <= pred_floor))
    q, n_excluded = qlike(actual, pred, pred_floor)
    m = mse(actual, pred)
    notes = []
    if n_clipped:
        notes.append(f"{n_clipped} predictions clipped to the floor")
    if n_excluded:
        notes.append(f"{n_excluded} zero-actual rows excluded from QLIKE")
    note = f" ({', '.join(notes)})" if notes else ""
    print(f"{name:>18s}  QLIKE={q:.6f}  MSE={m:.6e}{note}")
    return q, m


def main():
    dataset = build_dataset()
    train_mask, test_mask = split_dataset(dataset)

    dates = np.array(dataset["dates"])
    print(f"split date: {SPLIT_DATE}")
    print(f"train: {int(train_mask.sum())} rows, {dates[train_mask][0]}..{dates[train_mask][-1]}")
    print(f"test:  {int(test_mask.sum())} rows, {dates[test_mask][0]}..{dates[test_mask][-1]}")
    print()

    rv_train = dataset["rv"][train_mask]
    rv_test = dataset["rv"][test_mask]
    pred_floor = float(rv_train[rv_train > 0].min())
    print(f"QLIKE prediction floor: {pred_floor:.6e} (smallest nonzero realized-variance "
          f"proxy value in the training window, set by the data's return quantization)")
    print()

    r_train = dataset["market_return"][train_mask]
    r_test = dataset["market_return"][test_mask]
    omega, alpha, beta, opt_result = fit_garch(r_train)
    print(f"GARCH(1,1) fit: omega={omega:.6e} alpha={alpha:.6f} beta={beta:.6f} "
          f"(alpha+beta={alpha + beta:.6f}), converged={opt_result.success}")
    garch_pred = rolling_forecast(omega, alpha, beta, r_train, r_test)

    har_beta = fit_har(dataset, train_mask)
    print(f"HAR-RV fit: intercept={har_beta[0]:.6e} daily={har_beta[1]:.4f} "
          f"weekly={har_beta[2]:.4f} monthly={har_beta[3]:.4f}")
    har_pred = predict_har(dataset, test_mask, har_beta)

    naive_pred = naive_forecast(dataset, test_mask)

    print()
    report("GARCH(1,1)", rv_test, garch_pred, pred_floor)
    report("HAR-RV", rv_test, har_pred, pred_floor)
    report("naive persistence", rv_test, naive_pred, pred_floor)


if __name__ == "__main__":
    main()
