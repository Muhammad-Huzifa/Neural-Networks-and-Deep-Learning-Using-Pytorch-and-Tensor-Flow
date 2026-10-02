# Development

Use a fresh kernel and the environment documented by the notebook section. PyTorch and TensorFlow requirements are separate. The mixed CIFAR example needs both frameworks.

```bash
python scripts/check_notebooks.py
python projects/object_detection/yolo/predict.py --help
python projects/object_detection/yolo/train.py --help
```

These commands check active source structure and CLI arguments. They do not install frameworks or perform GPU training and detector inference. Keep custom datasets, credentials, and generated checkpoints out of Git. Classical ML and Adult Income now belong in the separate Machine Learning collection.
