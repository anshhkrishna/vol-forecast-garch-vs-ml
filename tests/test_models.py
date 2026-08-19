import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from baseline import qlike
from data import build_dataset
from har import fit_har, predict_har
from ml_model import fit_ml, ml_features, predict_ml
from rigor import (
    SEEDS,
    check_no_lookahead,
    ml_qlike_sweep,
    qlike_contributions,
)
from split import split_dataset


def test_no_lookahead_in_features():
    dataset = build_dataset()
    n_checked = check_no_lookahead(dataset)
    assert n_checked == int(dataset["valid"].sum())


def test_ml_model_does_not_beat_har_rv_on_mean_qlike():
    dataset = build_dataset()
    train_mask, test_mask = split_dataset(dataset)

    rv_train = dataset["rv"][train_mask]
    rv_test = dataset["rv"][test_mask]
    pred_floor = float(rv_train[rv_train > 0].min())

    har_beta = fit_har(dataset, train_mask)
    har_pred = predict_har(dataset, test_mask, har_beta)
    har_q, _ = qlike(rv_test, har_pred, pred_floor)

    qlikes = ml_qlike_sweep(dataset, train_mask, test_mask, pred_floor, seeds=SEEDS)
    mean_q = float(np.mean(qlikes))

    assert len(qlikes) >= 5
    assert mean_q > har_q


def test_ml_qlike_seed_variance_is_small():
    dataset = build_dataset()
    train_mask, test_mask = split_dataset(dataset)

    rv_train = dataset["rv"][train_mask]
    pred_floor = float(rv_train[rv_train > 0].min())

    qlikes = ml_qlike_sweep(dataset, train_mask, test_mask, pred_floor, seeds=SEEDS)
    std_q = float(np.std(qlikes))

    assert std_q < 1.0


def test_ml_qlike_gap_is_concentrated_in_one_day():
    """The ML model's QLIKE gap to HAR-RV is one row, not a spread-out gap.

    Locks in the decomposition src/rigor.py prints: a single test day carries
    almost the whole ML QLIKE sum, and on every other day the model is
    competitive with HAR-RV. Without this, the headline average reads as a
    model-quality gap when it is a single clipped prediction.
    """
    dataset = build_dataset()
    train_mask, test_mask = split_dataset(dataset)

    rv_train = dataset["rv"][train_mask]
    rv_test = dataset["rv"][test_mask]
    pred_floor = float(rv_train[rv_train > 0].min())

    ml_pred = predict_ml(fit_ml(dataset, train_mask, random_state=0), dataset, test_mask)
    ml_contrib, _ = qlike_contributions(rv_test, ml_pred, pred_floor)
    top = int(np.argmax(ml_contrib))
    assert ml_contrib[top] / ml_contrib.sum() > 0.99

    har_pred = predict_har(dataset, test_mask, fit_har(dataset, train_mask))
    har_contrib, _ = qlike_contributions(rv_test, har_pred, pred_floor)
    ml_without = float(np.delete(ml_contrib, top).mean())
    har_without = float(np.delete(har_contrib, top).mean())
    assert abs(ml_without - har_without) < 0.05


def test_ml_collapse_is_a_negative_prediction_not_a_range_ceiling():
    """The collapse day is an unclipped NEGATIVE prediction, not an inability
    to forecast above the training range.

    Squared-error boosting has no non-negativity constraint, so its additive
    score can go below zero; predict_ml clips that to 0 and qlike then floors
    it, which is what produces the enormous ratio. The training window's
    realized variance also reaches well above anything in the test window, so
    a range ceiling cannot be the explanation.
    """
    dataset = build_dataset()
    train_mask, test_mask = split_dataset(dataset)

    rv_train = dataset["rv"][train_mask]
    rv_test = dataset["rv"][test_mask]
    pred_floor = float(rv_train[rv_train > 0].min())

    model = fit_ml(dataset, train_mask, random_state=0)
    ml_pred = predict_ml(model, dataset, test_mask)
    ml_contrib, scored = qlike_contributions(rv_test, ml_pred, pred_floor)
    top_row = int(np.flatnonzero(scored)[int(np.argmax(ml_contrib))])

    raw = model.predict(ml_features(dataset, test_mask))
    assert raw[top_row] < 0.0
    assert raw.max() > float(predict_har(dataset, test_mask,
                                         fit_har(dataset, train_mask)).max())
    assert rv_test.max() <= rv_train.max()
