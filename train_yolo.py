from ultralytics import YOLO

model = YOLO("configs/yolo/dentex_yolov8x.yaml", task="detect").load("checkpoints/yolov8x.pt")
model.train(data="configs/yolo/dentex_disease_dataset.yaml", epochs=100)
