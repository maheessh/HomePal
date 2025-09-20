from ultralytics import YOLO

model = YOLO("yolov8s-pose.pt")
model.export(format="ncnn", imgsz=320)  # creates 'yolo11n_ncnn_model'