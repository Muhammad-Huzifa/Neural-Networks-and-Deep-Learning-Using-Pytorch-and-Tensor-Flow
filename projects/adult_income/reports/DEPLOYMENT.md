# Local model serving

Train the pipeline, install `requirements-apps.txt`, then run `python -m uvicorn apps.api:app --reload` or `python -m streamlit run deployment/app.py` from the project root.

The API model path can be set through the `ADULT_MODEL_PATH` environment variable. API startup loads the model once and reports `model_ready`; prediction requests return 503 when it is missing.

A hosted service needs the generated model artifact and the same dependency environment as training. Hosting is not configured or performed by this repository refactor.
