# vol-forecast-garch-vs-ml

> garch(1,1), har-rv, and gradient boosting compared out of sample on daily realized variance, in numpy and scikit-learn

## at a glance

| model | qlike (out of sample) | mse (out of sample) |
|---|---|---|
| garch(1,1) | 1.522779 | 1.648851e-07 |
| har-rv | 1.516278 | 1.603824e-07 |
| naive persistence | 226.250424 | 2.446041e-07 |
| gradient boosting (mean of 5 seeds, std 0.000039) | 197.878876 | 1.817569e-07 |

(`results/baseline.log` lines 10 to 12, `results/rigor.log` line 11.) the claim under
test: a gradient-boosted regressor given the identical daily/weekly/monthly
realized-variance feature window as har-rv does not beat har-rv out of sample, and it
does not, by a wide margin.

## how it was measured

four one-day-ahead forecasters for realized variance, scored on the identical
out-of-sample rows:

- **garch(1,1)**, `sigma_t^2 = omega + alpha*r_{t-1}^2 + beta*sigma_{t-1}^2`, fit by
  gaussian quasi-maximum likelihood via `scipy.optimize.minimize` (l-bfgs-b) on the
  training window, then rolled forward through the test window one step at a time.
- **har-rv**, ordinary least squares (closed-form normal equations, no
  `sklearn.linear_model`) regressing next-day realized variance on trailing daily,
  weekly, and monthly realized-variance averages, the standard corsi specification.
- **naive persistence**, tomorrow's forecast is today's realized variance.
- **gradient boosting**, `sklearn.ensemble.GradientBoostingRegressor` (scikit-learn's
  own default hyperparameters, not tuned) trained on the identical daily/weekly/monthly
  feature window har-rv uses, evaluated across 5 fixed seeds (`results/rigor.log` lines 5 to 9), no retraining inside the test window.

data is `data/F-F_Research_Data_Factors_daily.csv`, the fama/french daily factor file
(see `data/README.md`). the daily market return is `Mkt-RF + RF`, converted from
percent to a decimal; realized variance is its square, a squared-daily-return proxy
rather than a true intraday-sampled realized volatility, since no intraday prices are
vendored. all four forecasters are scored on this same proxy.

the split is fixed by date and set before any model was fit: train through `20091231`
(22105 rows), test from `20100104` onward (4147 rows, `results/baseline.log` lines 1 to
3). the split is not touched again after being chosen.

qlike (`actual/pred - log(actual/pred) - 1`) is the headline loss, since it is
scale-free and penalizes under-prediction of variance harder than over-prediction, the
property that matters for a risk forecast. mse is reported alongside as a familiar
cross-check.

## caveats

- **the realized-variance target is a squared-daily-return proxy**, not the
  intraday-sampled realized volatility har-rv was designed around. this narrows har-rv's
  usual literature edge over garch(1,1), since both models are working from the same
  coarse daily signal here rather than har-rv getting the richer high-frequency input it
  normally uses.
- **qlike is undefined at exact-zero realized variance.** 31 out-of-sample days have a
  market return that rounds to exactly zero in the vendored file; those rows are
  excluded from every model's qlike average (not from mse), and the excluded count is
  printed in `results/baseline.log` and `results/run.log`.
- **the gap is not evenly spread.** gradient boosting's qlike sits close to naive
  persistence's rather than anywhere near garch(1,1) or har-rv's (see the table above),
  consistent with a tree ensemble having no way to predict a realized-variance value
  outside the range it saw in training, unlike garch's multiplicative recursion or
  har-rv's linear extrapolation. the training window predates the market-stress episodes
  present in the test window. a programmatic check confirms this is not a lookahead bug:
  no feature at row t reaches into its own or a later target (`results/rigor.log` line
  1).
- **no seed dependence worth reporting.** across 5 fixed seeds, gradient boosting's
  qlike varies by 0.000039 around a mean of 197.878876 (`results/rigor.log` lines 5 to
  11), five orders of magnitude smaller than the gap to har-rv. a different random state
  would not flip this result.
- **scale.** this tests one feature set (the har-rv trailing-average window) and one
  series (a broad daily equity market return). it says nothing about whether gradient
  boosting does better with features har-rv does not use (implied volatility, order
  flow, jump components), which is where the literature on ml-vs-har-rv comparisons
  reports most of its gains coming from, or on individual equities rather than a market
  aggregate.

## reproduce

set up the pinned environment first, since garch's optimizer needs `scipy` alongside
`numpy` and `scikit-learn`.

```
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

run the three reference baselines on their own to see garch and har-rv land close
together while naive persistence does not.

```
python src/baseline.py
```

run the full four-way out-of-sample comparison, including the gradient-boosting model,
which is what `results/run.log` was captured from.

```
python src/experiment.py
```

reproduce the 5-seed sweep and the no-lookahead check behind the caveats above.

```
python src/rigor.py
```

run the test suite, including the test that asserts the core claim in the direction the
data actually shows it.

```
python -m pytest tests/ -v
```

regenerate `results/headline.png` from the committed logs.

```
python src/plot.py
```
