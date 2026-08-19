"""Seed sweep for the ML model's out-of-sample QLIKE, a feature leakage check,
and a per-day decomposition of where each forecaster's QLIKE actually comes from.

Fits the gradient-boosted model across several random seeds to separate
genuine seed variance from any single-day prediction collapse, checks
programmatically that no daily/weekly/monthly feature reaches into its own
or a later target, and then breaks the headline QLIKE averages down day by
day, since a mean over 4116 days says nothing about whether the gap between
two forecasters is spread across the window or concentrated in one row.
"""

import numpy as np

from baseline import qlike
from data import build_dataset
from garch import fit_garch, rolling_forecast
from har import fit_har, naive_forecast, predict_har
from ml_model import fit_ml, ml_features, predict_ml
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


def qlike_contributions(actual, pred, pred_floor):
    """Per-day QLIKE terms, on the same rows the headline average is taken over.

    Returns (contributions, scored_mask). `scored_mask` marks the rows with a
    strictly positive actual, the ones `baseline.qlike` averages over; the
    contributions line up with those rows in order, so contributions.mean()
    equals the reported QLIKE exactly.
    """
    pred = np.maximum(pred, pred_floor)
    scored = actual > 0
    ratio = actual[scored] / pred[scored]
    return ratio - np.log(ratio) - 1.0, scored


def describe_concentration(name, actual, pred, pred_floor, dates):
    """Prints where `name`'s QLIKE sum comes from, and what its mean would be
    without its single worst day. Returns (top_date, mean_excluding_top).
    """
    contributions, scored = qlike_contributions(actual, pred, pred_floor)
    scored_dates = dates[scored]
    total = float(contributions.sum())
    top = int(np.argmax(contributions))
    share = 100.0 * contributions[top] / total
    without_top = float(np.delete(contributions, top).mean())
    floored = np.maximum(pred, pred_floor)[scored]
    print(f"  {name:>22s}  worst day {scored_dates[top]} contributes "
          f"{contributions[top]:.1f} of {total:.1f} ({share:.2f}% of the sum), "
          f"actual={actual[scored][top]:.6e} predicted={floored[top]:.6e}")
    print(f"  {'':>22s}  mean QLIKE over the other {len(contributions) - 1} days: "
          f"{without_top:.6f}")
    return scored_dates[top], without_top


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

    print()
    n_scored = int(np.sum(rv_test > 0))
    print(f"per-day QLIKE decomposition over the {n_scored} scored days (seed 0, the "
          f"configuration run.log was captured from)")
    dates = np.array(dataset["dates"])[test_mask]

    r_train = dataset["market_return"][train_mask]
    r_test = dataset["market_return"][test_mask]
    omega, alpha, beta, _ = fit_garch(r_train)
    garch_pred = rolling_forecast(omega, alpha, beta, r_train, r_test)
    naive_pred = naive_forecast(dataset, test_mask)
    ml_model = fit_ml(dataset, train_mask, random_state=0)
    ml_pred = predict_ml(ml_model, dataset, test_mask)

    describe_concentration("GARCH(1,1)", rv_test, garch_pred, pred_floor, dates)
    describe_concentration("HAR-RV", rv_test, har_pred, pred_floor, dates)
    describe_concentration("naive persistence", rv_test, naive_pred, pred_floor, dates)
    ml_top_date, ml_without_top = describe_concentration(
        "ML (gradient boosting)", rv_test, ml_pred, pred_floor, dates)

    har_contrib, _ = qlike_contributions(rv_test, har_pred, pred_floor)
    har_top_idx = int(np.argmax(har_contrib))
    ml_contrib, scored = qlike_contributions(rv_test, ml_pred, pred_floor)
    ml_top_idx = int(np.argmax(ml_contrib))
    har_without_ml_top = float(np.delete(har_contrib, ml_top_idx).mean())
    print()
    print(f"excluding only {ml_top_date}, on the identical remaining "
          f"{len(ml_contrib) - 1} days: ML mean QLIKE={ml_without_top:.6f}, "
          f"HAR-RV mean QLIKE={har_without_ml_top:.6f}")

    raw_ml = ml_model.predict(ml_features(dataset, test_mask))
    n_negative = int(np.sum(raw_ml < 0))
    top_raw = float(raw_ml[np.flatnonzero(scored)[ml_top_idx]])
    print(f"ML raw prediction on {ml_top_date} before the non-negativity clip: "
          f"{top_raw:.6e} (negative raw predictions across the whole test window: "
          f"{n_negative})")
    print(f"ML raw prediction range on the test window: [{raw_ml.min():.6e}, "
          f"{raw_ml.max():.6e}]")

    rv_train_max = float(rv_train.max())
    rv_test_max = float(rv_test.max())
    n_above = int(np.sum(rv_test > rv_train_max))
    n_train_at_least = int(np.sum(rv_train >= rv_test[np.flatnonzero(scored)[ml_top_idx]]))
    print(f"training realized-variance max={rv_train_max:.6e}, test max={rv_test_max:.6e}, "
          f"test days above the training max: {n_above}")
    print(f"training days whose realized variance is at least {ml_top_date}'s actual: "
          f"{n_train_at_least}")


if __name__ == "__main__":
    main()
