import numpy as np
import pandas as pd

NUMERIC = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "country"]
FEATURES = NUMERIC + CATEGORICAL
CSV_COLUMNS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "country", "income"]

def prepare_features(frame):
    frame = frame.copy()
    frame.columns = [str(c).strip().replace("_", "-") for c in frame.columns]
    frame = frame.rename(columns={"native-country": "country"})
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError("Missing input columns: " + ", ".join(sorted(missing)))
    frame = frame[FEATURES].copy()
    for column in NUMERIC:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    for column in CATEGORICAL:
        frame[column] = frame[column].map(lambda value: str(value).strip() if pd.notna(value) else np.nan)
        frame[column] = frame[column].replace({"?": np.nan, "": np.nan})
    return frame

def load_dataset(path):
    frame = pd.read_csv(path, skipinitialspace=True)
    if "income" not in frame.columns:
        raise ValueError("CSV must have a header and an income column. Run scripts/download_data.py or consult data/README.md.")
    target = frame["income"].astype(str).str.strip().str.rstrip(".").map({"<=50K": 0, ">50K": 1})
    if target.isna().any() or target.nunique() != 2:
        raise ValueError("Income labels must contain both <=50K and >50K classes.")
    return prepare_features(frame), target.astype(int)
