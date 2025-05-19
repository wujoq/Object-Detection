from ultralytics import YOLO

model = YOLO("yolo_training/tool_detector/weights/best.pt")
model.export(format="onnx")