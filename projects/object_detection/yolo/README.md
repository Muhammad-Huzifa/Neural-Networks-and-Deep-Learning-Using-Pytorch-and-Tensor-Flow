# YOLO detection starter

Small Python launchers for Ultralytics training, validation, and inference. Install from the collection root:

```bash
python -m pip install -r requirements/detection.txt
```

Predict from the collection root:

```bash
python projects/object_detection/yolo/predict.py --model yolov8n.pt --source path/to/image.jpg
```

The framework downloads the named pretrained checkpoint when it is absent. To use a custom detector, pass its local checkpoint path instead. No checkpoint is bundled.

For training, prepare images and YOLO label files, copy `configs/dataset.example.yaml` to a local YAML file, then set its actual dataset path and class names:

```bash
python projects/object_detection/yolo/train.py --data path/to/dataset.yaml --model yolov8n.pt --epochs 50
python projects/object_detection/yolo/train.py --data path/to/dataset.yaml --model path/to/best.pt --evaluate
```

Default generated results are saved under this project's `runs/` directory and ignored by Git. CLI argument handling has been checked; real model inference, dataset validation, and training require Ultralytics and data and have not been run during this restructuring.

See [Ultralytics Python usage](https://docs.ultralytics.com/usage/python/) for supported models and dataset conventions.
