import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import MISSING_CODES, build_dataset, parse_factors_csv


def test_parsed_date_range_matches_file():
    dates, _, _ = parse_factors_csv()
    assert dates[0] == "19260701"
    assert dates[-1] == "20260630"
    assert len(dates) == 26274


def test_no_missing_value_codes_remain():
    _, mkt_rf, rf = parse_factors_csv()
    for code in MISSING_CODES:
        assert not np.any(mkt_rf == code)
        assert not np.any(rf == code)


def test_rv_non_negative_and_finite_after_warm_up():
    dataset = build_dataset()
    rv = dataset["rv"][dataset["valid"]]
    assert np.all(np.isfinite(rv))
    assert np.all(rv >= 0)
    for col in ("daily_rv", "weekly_rv", "monthly_rv"):
        values = dataset[col][dataset["valid"]]
        assert np.all(np.isfinite(values))
        assert np.all(values >= 0)


def test_weekly_and_monthly_features_match_hand_computed_window():
    dataset = build_dataset()
    rv = dataset["rv"]

    t = 5000
    expected_weekly = np.mean(rv[t - 5:t])
    expected_monthly = np.mean(rv[t - 22:t])
    assert dataset["weekly_rv"][t] == expected_weekly
    assert dataset["monthly_rv"][t] == expected_monthly
    assert dataset["daily_rv"][t] == rv[t - 1]


def test_features_at_row_t_use_no_data_at_or_after_t():
    dataset = build_dataset()
    rv = dataset["rv"]
    valid_idx = np.where(dataset["valid"])[0]

    t = valid_idx[100]
    manual_weekly = np.mean(rv[t - 5:t])
    manual_monthly = np.mean(rv[t - 22:t])
    assert dataset["weekly_rv"][t] == manual_weekly
    assert dataset["monthly_rv"][t] == manual_monthly
