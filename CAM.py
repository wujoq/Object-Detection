import cv2
import torch
import numpy as np
from ultralytics import YOLO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision.transforms import ToTensor
import sys

# === Konfiguracja ===
image_path = "dataset/test/images/composite_pexels-andrejcook-131723_04.png"  # Podmień na swój obraz testowy
model_path = "yolo_training/tool_detector/weights/best.pt"
class_index = 2  # 0: driver, 1: komb, 2: sword
output_path = "cam_output.png"

# === Wczytaj model YOLO ===
model = YOLO(model_path)
model.fuse()
model.model.eval()

# === Wczytaj i przeskaluj obraz ===
resize_dim = 640
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
resized_image = cv2.resize(image_rgb, (resize_dim, resize_dim))
resized_image_norm = resized_image.astype(np.float32) / 255.0
input_tensor = ToTensor()(resized_image_norm).unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# === Wybierz warstwę docelową ===
try:
    target_layers = [model.model.model[-2]]
except Exception as e:
    print("❌ Nie można znaleźć warstwy docelowej do CAM:", e)
    sys.exit(1)

# === Grad-CAM z użyciem use_cuda ===
cam = GradCAM(model=model.model, target_layers=target_layers)
targets = [ClassifierOutputTarget(class_index)]

# === Wygeneruj CAM ===
input_tensor.requires_grad_()  # Wymagane do uzyskania gradientów

try:
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    visualization = show_cam_on_image(resized_image_norm, grayscale_cam, use_rgb=True)

    # === Zapisz wynik ===
    cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"✅ CAM zapisany do: {output_path}")
except Exception as e:
    print(f"❌ Błąd podczas generowania CAM: {e}")
