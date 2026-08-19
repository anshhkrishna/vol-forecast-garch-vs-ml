import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from baseline import qlike
from data import build_dataset
from har import fit_har, predict_har
from rigor import SEEDS, check_no_lookahead, ml_qlike_sweep
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
