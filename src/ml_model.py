"""Gradient-boosted regression on the HAR-RV feature window.

Trains sklearn.ensemble.GradientBoostingRegressor on the same
daily/weekly/monthly trailing realized-variance features HAR-RV uses, fit
once on the training split and never retrained inside the test window.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

N_ESTIMATORS = 100
MAX_DEPTH = 3
LEARNING_RATE = 0.1


def ml_features(dataset, mask):
    """Builds the [daily_rv, weekly_rv, monthly_rv] design matrix for the
    rows selected by `mask` -- the same feature window HAR-RV uses, without
    HAR's intercept column (gradient boosting doesn't need one).
    """
    daily = dataset["daily_rv"][mask]
    weekly = dataset["weekly_rv"][mask]
    monthly = dataset["monthly_rv"][mask]
    return np.column_stack([daily, weekly, monthly])


def fit_ml(dataset, train_mask, random_state=0):
    """Fits GradientBoostingRegressor on the training rows only."""
    X_train = ml_features(dataset, train_mask)
    y_train = dataset["rv"][train_mask]
    model = GradientBoostingRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def predict_ml(model, dataset, mask):
    """Predicts realized variance for the rows selected by `mask`.

    Gradient-boosted regression has no built-in non-negativity constraint,
    so predictions are clipped at zero before being returned.
    """
    X = ml_features(dataset, mask)
    preds = model.predict(X)
    return np.clip(preds, 0.0, None)


if __name__ == "__main__":
    from data import build_dataset
    from split import split_dataset

    dataset = build_dataset()
    train_mask, test_mask = split_dataset(dataset)

    train_idx = np.flatnonzero(train_mask)[:500]
    small_train_mask = np.zeros_like(train_mask)
    small_train_mask[train_idx] = True

    model = fit_ml(dataset, small_train_mask)
    preds = predict_ml(model, dataset, test_mask)
    assert np.all(np.isfinite(preds))
    assert np.all(preds >= 0.0)
    print(f"trained on {int(small_train_mask.sum())} rows, predicted {len(preds)} test rows")
    print(f"prediction range: [{preds.min():.6e}, {preds.max():.6e}]")
