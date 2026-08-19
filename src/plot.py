"""Regenerates results/headline.png from the committed results/baseline.log and
results/rigor.log -- parses the real QLIKE numbers those steps already printed, rather
than recomputing anything, so the chart always matches exactly what is checked in.

Run with `python src/plot.py`.
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
BASELINE_LOG = RESULTS_DIR / "baseline.log"
RIGOR_LOG = RESULTS_DIR / "rigor.log"
OUT_PATH = RESULTS_DIR / "headline.png"

BASELINE_ROW_RE = re.compile(
    r"^\s*(GARCH\(1,1\)|HAR-RV|naive persistence)\s+QLIKE=([\d.]+)\s+MSE="
)
RIGOR_MEAN_RE = re.compile(
    r"^ML \(gradient boosting\) QLIKE across \d+ seeds: mean=([\d.]+) std=([\d.]+)"
)


def parse_baseline_qlike(path=BASELINE_LOG):
    """Returns {label: qlike} for GARCH(1,1), HAR-RV, and naive persistence."""
    values = {}
    for line in path.read_text().splitlines():
        m = BASELINE_ROW_RE.match(line)
        if m:
            values[m.group(1)] = float(m.group(2))
    assert len(values) == 3, f"expected 3 baseline rows, parsed {values}"
    return values


def parse_ml_qlike(path=RIGOR_LOG):
    """Returns (mean, std) QLIKE for the gradient boosting model across seeds."""
    for line in path.read_text().splitlines():
        m = RIGOR_MEAN_RE.match(line)
        if m:
            return float(m.group(1)), float(m.group(2))
    raise AssertionError("no ML seed-sweep summary line found in rigor.log")


def plot(baseline_qlike, ml_mean, ml_std, out_path=OUT_PATH):
    labels = ["garch(1,1)", "har-rv", "naive\npersistence", "ml (gradient\nboosting)"]
    values = [baseline_qlike["GARCH(1,1)"], baseline_qlike["HAR-RV"],
              baseline_qlike["naive persistence"], ml_mean]
    errors = [0, 0, 0, ml_std]
    colors = ["#1f77b4", "#2ca02c", "#7f7f7f", "#d62728"]

    fig, ax = plt.subplots(figsize=(1600 / 150, 900 / 150), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars = ax.bar(labels, values, yerr=errors, capsize=6, color=colors, width=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("out-of-sample qlike (log scale, lower is better)", fontsize=13)
    ax.set_title(
        "gradient boosting collapses toward naive persistence,\n"
        "while garch and har-rv stay flat and close together",
        fontsize=15,
    )
    ax.tick_params(labelsize=12)
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)

    for bar, v in zip(bars, values):
        ax.annotate(f"{v:.3g}", xy=(bar.get_x() + bar.get_width() / 2, v),
                    xytext=(0, 6), textcoords="offset points", ha="center", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    baseline_qlike = parse_baseline_qlike()
    ml_mean, ml_std = parse_ml_qlike()
    plot(baseline_qlike, ml_mean, ml_std)
