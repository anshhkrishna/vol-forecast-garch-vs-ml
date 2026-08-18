# Data

`F-F_Research_Data_Factors_daily.csv` is the Fama/French daily 3-factor file (Mkt-RF,
SMB, HML, RF), published by Kenneth R. French's data library at Dartmouth's Tuck School
of Business. Spans 1926-07-01 through the most recent month end. Returns are in percent.

`Mkt-RF + RF` reconstructs the daily total market return, whose squared value is used
here as the realized-variance proxy.

`MANIFEST.tsv` records the file's size, SHA-256 checksum, original source URL, and
retrieval date.

## Reading the file

Fixed-width-ish CSV with a three-line prose header, then a blank line, then the header
row (`,Mkt-RF,SMB,HML,RF`), then one row per trading day as `YYYYMMDD,val,val,val,val`,
terminated by a blank line and a copyright notice. `-99.99` is the missing-value code.

## Provenance

Published free for research use by Eugene F. Fama and Kenneth R. French. Vendored here
unmodified, with source URL and retrieval date in `MANIFEST.tsv`.
