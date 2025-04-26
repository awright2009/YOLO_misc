from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
#model = YOLO("yolo11n.pt")
model = YOLO("yolo11x-seg.pt")

from pathlib import Path
import os
import random
import cv2
import numpy as np


img = cv2.imread("right.JPG")

# if you want all classes
yolo_classes = list(model.names.values())
classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]
conf = 0.2
results = model.predict(img, conf=conf)

colors = [random.choices(range(256), k=3) for _ in classes_ids]

#i = 0
#for x in classes_ids:
#    print("class ", x, " has color ", colors[i])
#    print("class name is ", yolo_classes[i])
#    i = i + 1

for result in results:
    for box in result.boxes:
        color_number = classes_ids.index(int(box.cls[0]))
        print("class ", color_number, " has box ", box.xyxy.tolist())
        color_number = classes_ids.index(int(box.cls[0]))
        print("class ", color_number, " has color ", colors[color_number]);
        print("class name is ", yolo_classes[color_number])


for result in results:
    for mask in result.masks:
        print("mask ", box.xy.tolist())




for result in results:
    for mask, box in zip(result.masks.xy, result.boxes):
        points = np.int32([mask])
        color_number = classes_ids.index(int(box.cls[0]))
        cv2.fillPoly(img, points, colors[color_number])

cv2.imwrite("right_detect.jpg", img)

