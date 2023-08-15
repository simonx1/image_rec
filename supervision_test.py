import supervision as sv
from ultralytics import YOLO
from PIL import Image

img = '/Users/szymonkurcab/Desktop/dogs_n_cats.png'
image = Image.open(img).convert("RGB")

model = YOLO('yolov8s.pt')
result = model(image)[0]
detections = sv.Detections.from_yolov8(result)

len(detections)
