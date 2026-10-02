# Adult dataset

Source: [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult). Download the official training rows with `python scripts/download_data.py`; the downloader creates `adult.csv` with column headers. Network access is required. This refactor does not include a downloaded dataset.

The input CSV needs the fourteen Adult feature columns and an `income` column containing `<=50K` and `>50K`. `native-country` is accepted as an alias for `country`, and underscores are accepted in place of hyphens. Leading category spaces, question-mark missing values, and trailing periods in income labels are normalized.

The default training command creates a stratified 80/20 split of the downloaded `adult.data` rows. It does not evaluate the official separate `adult.test` file; record that distinction when comparing results with other experiments.

`sample_request.json` is an illustrative input record for the CLI and API, not a labelled evaluation example.

Dataset attribution: Barry Becker and Ronny Kohavi (1996), Adult, UCI Machine Learning Repository, DOI 10.24432/C5XW20. The dataset is distributed under CC BY 4.0.
