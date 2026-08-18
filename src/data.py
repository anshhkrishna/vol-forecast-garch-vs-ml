"""Loads the Fama/French daily factor file and builds HAR-RV features.

The realized-variance proxy used throughout this project is the square of the
daily total market return (Mkt-RF + RF), since the vendored data has no
intraday prices to build a true high-frequency realized variance from.
"""

from pathlib import Path

import numpy as np

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "F-F_Research_Data_Factors_daily.csv"

MISSING_CODES = {-99.99, -999.0}

WEEKLY_WINDOW = 5
MONTHLY_WINDOW = 22


def parse_factors_csv(path=DATA_PATH):
    """Parses the Ken French daily 3-factor CSV into (dates, mkt_rf, rf).

    Skips the three-line prose header and the blank line before the column
    header row, and stops at the trailing blank line before the copyright
    notice. Returns dates as an array of 'YYYYMMDD' strings and mkt_rf/rf as
    float arrays in percent, as published.
    """
    lines = Path(path).read_text().splitlines()

    header_idx = next(i for i, line in enumerate(lines) if line.startswith(",Mkt-RF"))

    dates = []
    mkt_rf = []
    rf = []
    for line in lines[header_idx + 1:]:
        line = line.strip()
        if not line:
            break
        parts = [p.strip() for p in line.split(",")]
        if not parts[0].isdigit():
            break
        dates.append(parts[0])
        mkt_rf.append(float(parts[1]))
        rf.append(float(parts[4]))

    mkt_rf = np.array(mkt_rf, dtype=np.float64)
    rf = np.array(rf, dtype=np.float64)

    for name, arr in (("Mkt-RF", mkt_rf), ("RF", rf)):
        for code in MISSING_CODES:
            if np.any(arr == code):
                raise ValueError(f"missing-value code {code} found in column {name}")

    return dates, mkt_rf, rf


def trailing_mean(values, window):
    """Trailing mean of `values` over `window` days strictly before each index.

    Result[t] = mean(values[t-window:t]); the first `window` entries are NaN
    since they don't have a full window of prior history.
    """
    n = len(values)
    out = np.full(n, np.nan, dtype=np.float64)
    for t in range(window, n):
        out[t] = np.mean(values[t - window:t])
    return out


def build_dataset(path=DATA_PATH):
    """Builds the market return series, realized-variance proxy, and HAR-RV
    features (daily/weekly/monthly trailing realized variance).

    Every feature at row t is built only from data at or before t - 1: daily
    is rv[t-1], weekly is the mean of rv[t-5:t], monthly is the mean of
    rv[t-22:t]. Rows before the 22-day warm-up have NaN features.
    """
    dates, mkt_rf, rf = parse_factors_csv(path)

    market_return = (mkt_rf + rf) / 100.0
    rv = market_return ** 2

    daily_rv = np.full(len(rv), np.nan, dtype=np.float64)
    daily_rv[1:] = rv[:-1]

    weekly_rv = trailing_mean(rv, WEEKLY_WINDOW)
    monthly_rv = trailing_mean(rv, MONTHLY_WINDOW)

    warm_up = MONTHLY_WINDOW
    valid = np.zeros(len(rv), dtype=bool)
    valid[warm_up:] = True

    return {
        "dates": dates,
        "market_return": market_return,
        "rv": rv,
        "daily_rv": daily_rv,
        "weekly_rv": weekly_rv,
        "monthly_rv": monthly_rv,
        "valid": valid,
    }


if __name__ == "__main__":
    dataset = build_dataset()
    n = len(dataset["dates"])
    n_valid = int(dataset["valid"].sum())
    print(f"parsed {n} rows, {dataset['dates'][0]}..{dataset['dates'][-1]}")
    print(f"{n_valid} rows with a full 22-day feature window")
