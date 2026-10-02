# Generated artifacts

`python train.py` writes `pipeline.joblib` and `metrics.json` here. The saved pipeline includes imputation, scaling, learned category encoding, and the classifier. The API, CLI, and interface all use this same pipeline.

Artifacts and downloaded data are ignored by Git. To share a trained run, include its package versions, dataset source, split definition, command, and metrics alongside the model download.
