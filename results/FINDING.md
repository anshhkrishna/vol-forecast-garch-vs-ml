A gradient-boosted regressor trained on the same daily/weekly/monthly realized-variance
feature window as HAR-RV, forecasting one-day-ahead realized variance on 4147
out-of-sample daily equity market return days, does not beat HAR-RV. Out-of-sample QLIKE:
HAR-RV 1.516278, GARCH(1,1) 1.522779, gradient boosting 197.878866 (mean across 5 seeds,
std 0.000051); out-of-sample MSE: HAR-RV 1.603824e-07, gradient boosting 1.817202e-07.
The surprising part is where that QLIKE gap lives. One day out of 4116, 2020-03-13,
carries 808186 of the 814469 total, 99.23 percent of it; excluding only that day,
gradient boosting scores 1.526779 against HAR-RV's 1.516339 on the identical remaining
days. So the model is roughly level with HAR-RV on an ordinary day and catastrophic on
one. On that day its raw prediction is negative, -2.173623e-04, the only negative
prediction in the test window: squared-error boosting sums trees with no non-negativity
constraint, the code clips that to zero, and QLIKE, which divides by the forecast, then
floors it at 1.000000e-08 against an actual of 8.082010e-03. It is not a failure to forecast
large values (its predictions reach 6.227674e-03, above HAR-RV's and GARCH's) and not an
out-of-range target (the training window's realized variance reaches 3.031081e-02 against
the test window's 1.440000e-02, with zero test days above the training maximum). Every
seed reproduces the same negative prediction on the same day.
