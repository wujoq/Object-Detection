import os
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

def log_results_to_tensorboard(log_dir):
    csv_path = os.path.join(log_dir, "results.csv")
    writer = SummaryWriter(log_dir)

    if not os.path.exists(csv_path):
        print(f"❌ Nie znaleziono pliku: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    for step, row in df.iterrows():
        if "train/box_loss" in row and not pd.isna(row["train/box_loss"]):
            writer.add_scalar("Loss/train_box", row["train/box_loss"], step)
        if "val/box_loss" in row and not pd.isna(row["val/box_loss"]):
            writer.add_scalar("Loss/val_box", row["val/box_loss"], step)

        if "train/cls_loss" in row:
            writer.add_scalar("Loss/train_cls", row["train/cls_loss"], step)
        if "val/cls_loss" in row:
            writer.add_scalar("Loss/val_cls", row["val/cls_loss"], step)

        if "metrics/mAP50(B)" in row:
            writer.add_scalar("Metric/mAP50", row["metrics/mAP50(B)"], step)

        if "metrics/precision(B)" in row:
            writer.add_scalar("Metric/precision", row["metrics/precision(B)"], step)

        if "metrics/recall(B)" in row:
            writer.add_scalar("Metric/recall", row["metrics/recall(B)"], step)

    writer.close()
    print(f"✅ Zapisano dane z {csv_path} do TensorBoarda")

if __name__ == "__main__":
    log_results_to_tensorboard("runs/train/exp_tensorboard")
