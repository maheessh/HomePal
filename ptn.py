from ultralytics import YOLO

model = YOLO("yolo11n_pose.pt")
model.export(format="ncnn", imgsz=320, int8=True)  # creates 'yolo11n_ncnn_model' with quantization