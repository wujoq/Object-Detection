from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import matplotlib.pyplot as plt

# Ustawienia
model_path = "yolo_training/tool_detector/weights/best.pt"
test_images_dir = "dataset/test/images"
test_labels_dir = "dataset/test/labels"
class_names = ["driver", "komb", "sword"]
img_size = 640

# Załaduj model
model = YOLO(model_path)

# Przygotuj listy wyników
y_true = []
y_pred = []

# Lista plików .png
image_files = sorted([f for f in os.listdir(test_images_dir) if f.endswith(".png")])

# Pętla przez testowe obrazy
for filename in image_files:
    image_path = os.path.join(test_images_dir, filename)
    label_path = os.path.join(test_labels_dir, filename.replace(".png", ".txt"))

    # Wczytaj obraz
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Predykcja
    results = model.predict(image_path, imgsz=img_size, conf=0.25, device="cuda", verbose=False)
    boxes = results[0].boxes

    # Pobrane predykcje
    pred_classes = boxes.cls.cpu().numpy().astype(int).tolist() if boxes is not None else []

    # Prawdziwe etykiety z pliku
    true_classes = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                class_id = int(line.strip().split()[0])
                true_classes.append(class_id)

    # Zbierz dane do metryk
    for cls in set(true_classes + pred_classes):
        y_true.append(int(cls in true_classes))
        y_pred.append(int(cls in pred_classes))

    # Wyświetl wynik z boxami
    annotated_frame = results[0].plot()
    plt.imshow(annotated_frame)
    plt.title(filename)
    plt.axis("off")
    plt.show()

# Oblicz metryki
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
print("\n Ewaluacja na zbiorze testowym:")
print(f" Accuracy:  {accuracy:.4f}")
print(f" Precision: {precision:.4f}")
print(f" Recall:    {recall:.4f}")
print(f" f1:    {f1:.4f}")


