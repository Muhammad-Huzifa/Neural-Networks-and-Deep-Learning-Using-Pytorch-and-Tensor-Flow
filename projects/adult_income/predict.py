import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from adult_income.model import predict_records

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description="Predict records from a JSON object or list of objects.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts/pipeline.joblib")
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error("The trained pipeline is missing. Run train.py first.")
    records = json.loads(args.input.read_text(encoding="utf-8"))
    records = records if isinstance(records, list) else [records]
    print(json.dumps(predict_records(joblib.load(args.model), pd.DataFrame(records)), indent=2))

if __name__ == "__main__":
    main()
