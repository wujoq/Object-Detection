from clearml import Task
import pandas as pd
import os

def log_metrics_from_csv(csv_path, task_name="YOLOv8 Metrics from CSV"):
    if not os.path.exists(csv_path):
        print(f"❌ Plik nie istnieje: {csv_path}")
        return

    # 1. Inicjalizacja eksperymentu
    task = Task.init(project_name="YOLO Experiments", task_name=task_name, task_type=Task.TaskTypes.testing)
    logger = task.get_logger()

    # 2. Wczytaj dane
    df = pd.read_csv(csv_path)

    # 3. Przejdź po wierszach i loguj metryki
    for epoch, row in df.iterrows():
        if "train/box_loss" in row:
            logger.report_scalar("Loss", "train_box", iteration=epoch, value=row["train/box_loss"])
        if "val/box_loss" in row:
            logger.report_scalar("Loss", "val_box", iteration=epoch, value=row["val/box_loss"])
        if "train/cls_loss" in row:
            logger.report_scalar("Loss", "train_cls", iteration=epoch, value=row["train/cls_loss"])
        if "val/cls_loss" in row:
            logger.report_scalar("Loss", "val_cls", iteration=epoch, value=row["val/cls_loss"])
        if "metrics/mAP50(B)" in row:
            logger.report_scalar("Metric", "mAP50", iteration=epoch, value=row["metrics/mAP50(B)"])
        if "metrics/precision(B)" in row:
            logger.report_scalar("Metric", "precision", iteration=epoch, value=row["metrics/precision(B)"])
        if "metrics/recall(B)" in row:
            logger.report_scalar("Metric", "recall", iteration=epoch, value=row["metrics/recall(B)"])

    print(f"✅ Zalogowano {len(df)} epok z {csv_path} do ClearML")
    task.close()

if __name__ == "__main__":
    # Ścieżka do pliku results.csv z Twojego treningu YOLO
    log_metrics_from_csv("yolo_training/tool_detector/results.csv")
