import argparse
import json
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from adult_income.data import load_dataset
from adult_income.model import build_pipeline

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description="Train and save the Adult Income preprocessing + classifier pipeline.")
    parser.add_argument("--data", type=Path, default=ROOT / "data/adult.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--model", choices=["gradient_boosting", "random_forest", "logistic"], default="gradient_boosting")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"Dataset is missing: {args.data}. Run python scripts/download_data.py first.")
    X, y = load_dataset(args.data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=args.seed, stratify=y)
    pipeline = build_pipeline(args.model, args.seed)
    pipeline.fit(X_train, y_train)
    predicted = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, list(pipeline.classes_).index(1)]
    metrics = {"model": args.model, "seed": args.seed, "train_rows": len(X_train), "test_rows": len(X_test), "accuracy": accuracy_score(y_test, predicted), "precision": precision_score(y_test, predicted, zero_division=0), "recall": recall_score(y_test, predicted, zero_division=0), "f1": f1_score(y_test, predicted, zero_division=0), "roc_auc": roc_auc_score(y_test, probabilities)}
    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.output / "pipeline.joblib")
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print("Saved", args.output / "pipeline.joblib")

if __name__ == "__main__":
    main()
