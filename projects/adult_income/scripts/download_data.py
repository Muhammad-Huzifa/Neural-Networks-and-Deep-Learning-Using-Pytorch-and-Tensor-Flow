import argparse
import io
from pathlib import Path
import urllib.request
import zipfile
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adult_income.data import CSV_COLUMNS

def main():
    parser = argparse.ArgumentParser(description="Download the official UCI Adult training data as a headered CSV.")
    parser.add_argument("--output", type=Path, default=ROOT / "data/adult.csv")
    args = parser.parse_args()
    url = "https://archive.ics.uci.edu/static/public/2/adult.zip"
    with urllib.request.urlopen(url, timeout=30) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        with archive.open("adult.data") as source:
            frame = pd.read_csv(source, names=CSV_COLUMNS, skipinitialspace=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print("Saved", len(frame), "rows to", args.output)

if __name__ == "__main__":
    main()
