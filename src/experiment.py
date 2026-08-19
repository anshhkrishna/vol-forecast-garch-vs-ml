"""Runs the full out-of-sample comparison: GARCH, HAR-RV, naive persistence,
and the gradient-boosted ML model, on the fixed split.
"""

import time

import numpy as np

from baseline import qlike, mse, report
from data import build_dataset
from garch import fit_garch, rolling_forecast
from har import fit_har, naive_forecast, predict_har
from ml_model import fit_ml, predict_ml
from split import SPLIT_DATE, split_dataset


def main():
    start = time.time()
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

    ml_fit_start = time.time()
    ml_model = fit_ml(dataset, train_mask)
    ml_pred = predict_ml(ml_model, dataset, test_mask)
    ml_fit_seconds = time.time() - ml_fit_start
    print(f"ML model: GradientBoostingRegressor trained on {int(train_mask.sum())} rows "
          f"in {ml_fit_seconds:.2f}s")

    print()
    garch_q, garch_m = report("GARCH(1,1)", rv_test, garch_pred, pred_floor)
    har_q, har_m = report("HAR-RV", rv_test, har_pred, pred_floor)
    naive_q, naive_m = report("naive persistence", rv_test, naive_pred, pred_floor)
    ml_q, ml_m = report("ML (gradient boosting)", rv_test, ml_pred, pred_floor)

    print()
    if ml_q < har_q:
        verdict = f"ML model BEATS HAR-RV on QLIKE ({ml_q:.6f} < {har_q:.6f})"
    elif ml_q > har_q:
        verdict = f"ML model DOES NOT beat HAR-RV on QLIKE ({ml_q:.6f} > {har_q:.6f})"
    else:
        verdict = f"ML model TIES HAR-RV on QLIKE ({ml_q:.6f} == {har_q:.6f})"
    print(verdict)

    elapsed = time.time() - start
    print(f"\nwall clock: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
