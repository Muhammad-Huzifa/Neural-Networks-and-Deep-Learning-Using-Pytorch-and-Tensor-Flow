# Adult Income Classification

An end-to-end tabular classification example using the UCI Adult dataset. Training saves one scikit-learn pipeline containing preprocessing and the classifier; the CLI, FastAPI application, and Streamlit interface use that pipeline for prediction.

## Quick start

Use Python 3.11 or 3.12. Run commands from the project root.

```bash
cd projects/adult_income
python -m venv .venv
```

Activate with `.venv\Scripts\activate.bat` in Windows Command Prompt, `.\.venv\Scripts\Activate.ps1` in PowerShell, `source .venv/Scripts/activate` in Windows Git Bash, or `source .venv/bin/activate` on Linux/macOS.

```bash
python -m pip install -r requirements.txt
python scripts/download_data.py
python train.py
python predict.py --input data/sample_request.json
```

The download requires internet access. To use an existing headered CSV, run `python train.py --data path/to/adult.csv`. See [the dataset guide](data/README.md) for the schema.

## Training and evaluation

```bash
python train.py --model gradient_boosting
python train.py --model logistic --output artifacts/logistic
python train.py --model random_forest --output artifacts/random_forest
```

Each run fits preprocessing on training rows and writes its held-out metrics to `metrics.json`. Model choices are explicit rather than selected using the test set. Generated models are not bundled. No new full-dataset benchmark result is claimed by this refactor.

## API and interface

```bash
python -m pip install -r requirements-apps.txt
python -m uvicorn apps.api:app --reload
```

Open http://127.0.0.1:8000/docs. `POST /predict` accepts the fields in [sample_request.json](data/sample_request.json). The API returns HTTP 503 until `artifacts/pipeline.joblib` exists.

In another terminal, from the project root:

```bash
python -m streamlit run deployment/app.py
```

The interface predicts locally and does not require the API to be running. The original `python deployment/07_api.py` launcher is retained.

## Structure

| Path | Purpose |
| --- | --- |
| `adult_income/` | Shared input normalization, preprocessing, and prediction |
| `train.py`, `predict.py` | Training and JSON prediction commands |
| `apps/` | FastAPI and Streamlit implementations |
| `deployment/` | Compatibility launchers |
| `scripts/download_data.py` | Official UCI dataset download |
| `data/`, `artifacts/` | Input instructions and generated files |
| `reports/` | Historical report and reproduction notes |
| `tests/` | Single-row, batch, unseen-category, and persistence checks |

## Validation

```bash
python -m unittest discover -s tests -v
```

Tests train a small synthetic fixture to verify preprocessing and prediction behavior. They do not measure performance on Adult. API/interface execution, external download, and full-dataset training must be checked in an environment with the optional dependencies and data.

See [reproduction notes](reports/REPRODUCIBILITY.md). Historical reported results are retained in [the original report](reports/PROJECT_REPORT.md) with an explicit provenance note.

Muhammad Huzifa — [GitHub](https://github.com/Muhammad-Huzifa)
