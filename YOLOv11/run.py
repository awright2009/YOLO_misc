from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
#model = YOLO("yolo11n.pt")
model = YOLO("yolo11x-seg.pt")

from pathlib import Path
import os

# Path to save the trained model
trained_model_path = "yolo11n_custom.pt"

# Check if the trained model already exists
if os.path.exists(trained_model_path):
    print(f"Loading trained model from: {trained_model_path}")
    model = YOLO(trained_model_path)
else:
    print("Training model from scratch...")
    model = YOLO("yolo11n.pt")  # Load pretrained base model
    results = model.train(data="coco8.yaml", epochs=100, imgsz=640)


    # Save the best trained model
    best_model_path = Path(results.save_dir) / "weights" / "best.pt"
    os.rename(best_model_path, trained_model_path)
    print(f"Trained model saved as: {trained_model_path}")


# Run inference with the YOLO11n model on the 'bus.jpg' image
results = model("left.JPG", save=True, project="output/", name="left_results")
results = model("right.JPG", save=True, project="output/", name="right_results")
