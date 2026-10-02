import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run Ultralytics detection on an image, video, directory, or camera source.")
    parser.add_argument("--model", default="yolov8n.pt", help="Checkpoint path or supported Ultralytics model name")
    parser.add_argument("--source", required=True)
    parser.add_argument("--conf", type=float, default=.25)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "runs")
    args = parser.parse_args()
    if not 0 <= args.conf <= 1:
        parser.error("Confidence must be between 0 and 1.")
    from ultralytics import YOLO
    source = int(args.source) if args.source.isdigit() else args.source
    YOLO(args.model).predict(source=source, conf=args.conf, save=True, project=str(args.output), name="predict")

if __name__ == "__main__":
    main()
