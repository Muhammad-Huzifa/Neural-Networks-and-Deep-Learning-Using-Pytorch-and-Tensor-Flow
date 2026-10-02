import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Train or evaluate a detector using an edited dataset YAML file.")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error("Dataset YAML does not exist. Prepare the dataset and edit the example first.")
    if args.epochs < 1 or args.imgsz < 1:
        parser.error("Epochs and image size must be positive.")
    from ultralytics import YOLO
    model = YOLO(args.model)
    if args.evaluate:
        model.val(data=str(args.data.resolve()), imgsz=args.imgsz)
    else:
        model.train(data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz, project=str(Path(__file__).resolve().parent / "runs"), name="train")

if __name__ == "__main__":
    main()
