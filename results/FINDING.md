A gradient-boosted regressor trained on the same daily/weekly/monthly realized-variance
feature window as HAR-RV, forecasting one-day-ahead realized variance on 4147
out-of-sample daily equity market return days, does not beat HAR-RV. Out-of-sample
QLIKE: HAR-RV 1.516278, GARCH(1,1) 1.522779, gradient boosting 197.878876 (mean across
5 seeds, std 0.000039). The gap is not close: the ML model's QLIKE lands near naive
persistence's 226.250424 rather than anywhere near GARCH or HAR-RV. The surprising part
is how little the result depends on chance: five different random seeds for the
regressor land within 0.000039 of each other, so this is not one unlucky training run.
The likely structural cause is that gradient-boosted trees cannot predict a target value
outside the range they saw during training, unlike GARCH's multiplicative recursion or
HAR-RV's linear extrapolation, and the test window contains realized-variance spikes
well above anything in almost a century of training data.
