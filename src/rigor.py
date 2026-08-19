"""Seed sweep for the ML model's out-of-sample QLIKE, plus a feature leakage check.

Fits the gradient-boosted model across several random seeds to separate
genuine seed variance from any single-day prediction collapse, and checks
programmatically that no daily/weekly/monthly feature reaches into its own
or a later target.
"""

import numpy as np

from baseline import qlike
from data import build_dataset
from har import fit_har, predict_har
from ml_model import fit_ml, predict_ml
from split import split_dataset

SEEDS = [0, 1, 2, 3, 4]


def check_no_lookahead(dataset):
    """Recomputes daily/weekly/monthly features directly from `rv` for every
    valid row and asserts they match the dataset's own values, confirming
    that no feature at row t uses rv[t] or anything after it.
    """
    rv = dataset["rv"]
    valid_idx = np.flatnonzero(dataset["valid"])
    for t in valid_idx:
        assert dataset["daily_rv"][t] == rv[t - 1]
        assert dataset["weekly_rv"][t] == np.mean(rv[t - 5:t])
        assert dataset["monthly_rv"][t] == np.mean(rv[t - 22:t])
    return len(valid_idx)


def ml_qlike_sweep(dataset, train_mask, test_mask, pred_floor, seeds=SEEDS):
    """Fits the ML model once per seed and returns the array of out-of-sample
    QLIKE values, one per seed.
    """
    rv_test = dataset["rv"][test_mask]
    qlikes = []
    for seed in seeds:
        model = fit_ml(dataset, train_mask, random_state=seed)
        pred = predict_ml(model, dataset, test_mask)
        q, _ = qlike(rv_test, pred, pred_floor)
        qlikes.append(q)
    return np.array(qlikes)


def main():
    dataset = build_dataset()
    train_mask, test_mask = split_dataset(dataset)

    n_checked = check_no_lookahead(dataset)
    print(f"leakage check: {n_checked} valid rows, no feature reaches into its own or a "
          f"later target")
    print()

    rv_train = dataset["rv"][train_mask]
    rv_test = dataset["rv"][test_mask]
    pred_floor = float(rv_train[rv_train > 0].min())

    har_beta = fit_har(dataset, train_mask)
    har_pred = predict_har(dataset, test_mask, har_beta)
    har_q, _ = qlike(rv_test, har_pred, pred_floor)
    print(f"HAR-RV QLIKE (fixed, no randomness): {har_q:.6f}")
    print()

    qlikes = ml_qlike_sweep(dataset, train_mask, test_mask, pred_floor)
    for seed, q in zip(SEEDS, qlikes):
        print(f"  seed={seed}  QLIKE={q:.6f}")
    mean_q = float(np.mean(qlikes))
    std_q = float(np.std(qlikes))
    print()
    print(f"ML (gradient boosting) QLIKE across {len(SEEDS)} seeds: "
          f"mean={mean_q:.6f} std={std_q:.6f}")
    print()
    if mean_q < har_q:
        verdict = f"ML model BEATS HAR-RV on mean QLIKE ({mean_q:.6f} < {har_q:.6f})"
    elif mean_q > har_q:
        verdict = f"ML model DOES NOT beat HAR-RV on mean QLIKE ({mean_q:.6f} > {har_q:.6f})"
    else:
        verdict = f"ML model TIES HAR-RV on mean QLIKE ({mean_q:.6f} == {har_q:.6f})"
    print(verdict)


if __name__ == "__main__":
    main()
