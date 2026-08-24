# Processed data (generated, not shipped)

This directory is the default `--out` target of `code/preprocess_nfv2.py`. It
is intentionally shipped empty in this repository — the four processed CSVs
it produces total ~1.1 GB, exceeding typical code-hosting file-size limits,
and are fully regenerable from the publicly available raw NFv2 datasets (see
the top-level `README.md`, "Raw data" and "Processed data" sections).

After running `bash code/batch_preprocess.sh` (or `preprocess_nfv2.py`
individually per dataset), this directory will contain:

```
NF-UNSW-NB15-v2_processed.csv
NF-UNSW-NB15-v2_feature_cols.txt
NF-BoT-IoT-v2_processed.csv
NF-BoT-IoT-v2_feature_cols.txt
NF-CSE-CIC-IDS2018-v2_processed.csv
NF-CSE-CIC-IDS2018-v2_feature_cols.txt
NF-ToN-IoT-v2_processed.csv
NF-ToN-IoT-v2_feature_cols.txt
```
