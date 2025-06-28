from ultralytics import YOLO

# model
model = YOLO('yolov8n.pt')

# Training the model
results = model.train(data="F:\computer_vision_leaerning\crowd_detection\crowd.yaml", epochs=20 ,imgsz=640)