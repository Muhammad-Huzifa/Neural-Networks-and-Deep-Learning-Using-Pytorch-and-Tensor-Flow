# Development and validation

## Source checks

```bash
python scripts/check_notebooks.py
```

This parses every active notebook and checks Python syntax. It does not execute original external-data or framework experiments.

## Self-contained lesson execution

Install `requirements/validation.txt`. In Deep Learning, also install CPU PyTorch with the command in the root README. Run from the repository root:

```bash
python scripts/execute_lessons.py
python scripts/execute_lessons.py --report artifacts/lesson_execution.json --output-dir artifacts/executed
```

The manifest `docs/NEW_LESSONS.json` selects the eight complete new lessons. Each runs in a fresh Python kernel using the command's interpreter, with a 180-second limit per cell and single-threaded numerical libraries. Failures stop execution. Source notebook outputs remain empty; optional reports/executed copies go into ignored `artifacts/`. Use `--only` with a manifest path for a focused rerun. Restart interactive kernels before validation.

GitHub Actions has a dedicated CPU lesson job that runs full Jupyter execution. Other checks retain their existing scope.

## Local validation — 2 October 2026

All eight new lessons / 40 code cells passed in fresh CPU Python processes with IPython display capture. This environment blocks Jupyter socket connections, so local execution used direct cell execution; the repository command and CI perform full kernel execution.

| Dependency | Local version |
| --- | --- |
| Python | 3.12.14 |
| numpy | 2.3.5 |
| pandas | 2.2.3 |
| scikit-learn | 1.8.0 |
| matplotlib | 3.10.8 |
| torch | 2.14.1+cpu |

Numerical checks cover all 33 manually differentiated parameters, causal attention under future-token perturbation, hand-calculated IoU, class-aware suppression, and checkpoint/logit equivalence. Five small PyTorch lessons perform actual CPU training with validation-based checkpoint selection; their scores apply only to their teaching data/splits.

YOLO source checks:

```bash
python projects/object_detection/yolo/predict.py --help
python projects/object_detection/yolo/train.py --help
```

These commands check arguments without loading a detector. Full TensorFlow/GPU experiments, custom image training, and real detector inference were not repeated. Preserve original dataset attributions and the retained binary archive. Keep credentials, private datasets, and generated checkpoints out of Git.
