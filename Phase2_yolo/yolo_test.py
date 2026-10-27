from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model("Phase2_yolo/bus.jpg")

results[0].show()