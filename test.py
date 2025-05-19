from ultralytics import YOLO
import os
import yaml

def main():
    # Paths
    base_dir = "dataset"
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")

    # 1. Create YOLOv8 YAML config
    data_yaml = {
        "path": base_dir,
        "train": "train/images",
        "val": "val/images",
        "names": ["driver", "komb", "sword"]
    }

    # Save the config
    os.makedirs("yolo_config", exist_ok=True)
    config_path = "yolo_config/dataset.yaml"
    with open(config_path, "w") as f:
        yaml.dump(data_yaml, f)

    # 2. Train the YOLOv8 model on GPU
    model = YOLO("yolov8n.yaml")
    model.train(
        data=config_path,
        epochs=100,
        imgsz=640,
        batch=16,
        device="cuda",
        project="yolo_training",
        name="tool_detector",
        exist_ok=True
    )

if __name__ == "__main__":
    main()
