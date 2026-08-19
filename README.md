# vol-forecast-garch-vs-ml

> garch(1,1), har-rv, and gradient boosting compared out of sample on daily realized variance, in numpy and scikit-learn

Status: scaffolded, not yet built.

## Claim under test

A gradient-boosted regression model, given the same lagged-realized-variance feature
window as HAR-RV, does not beat HAR-RV out of sample at one-day-ahead realized
volatility forecasting on a daily equity market return series. Any in-sample edge the
ML model shows is overfitting that does not survive the out-of-sample split.

## Baseline

Three reference forecasts, all evaluated out of sample on the same split:

- GARCH(1,1) fit by maximum likelihood.
- HAR-RV (linear regression on daily, weekly, and monthly realized-variance averages).
- Naive persistence: tomorrow's realized variance forecast is today's realized
  variance, the naive forecast most papers omit.

## Data

`data/F-F_Research_Data_Factors_daily.csv`, the Fama/French daily 3-factor file. The
daily market return series is built as `Mkt-RF + RF`, and squared daily returns are
used as the realized-variance proxy. See `data/README.md`.
