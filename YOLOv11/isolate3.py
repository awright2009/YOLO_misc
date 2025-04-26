from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
#model = YOLO("yolo11n.pt")
model = YOLO("yolo11x-seg.pt")

from pathlib import Path
import os
import random
import cv2
import numpy as np


filename = "right"

img = cv2.imread(filename + ".JPG")

depth = cv2.imread(filename + ".png", cv2.IMREAD_UNCHANGED)


# if you want all classes
yolo_classes = list(model.names.values())
classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]
conf = 0.2
results = model.predict(img, conf=conf)

colors = [random.choices(range(256), k=3) for _ in classes_ids]


i = 0

for r in results:
    img = np.copy(r.orig_img)
    depth = cv2.imread(filename + ".png", cv2.IMREAD_UNCHANGED)
    img_name = Path(r.path).stem

    # Iterate each object contour 


    for ci, c in enumerate(r):
        label = c.names[c.boxes.cls.tolist().pop()]

        b_mask = np.zeros(img.shape[:2], np.uint8)

        # Create contour mask 

        scale_percent = 0.8  # Scale by 80%

        # Get the contour
        contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)

        # Calculate centroid of the contour
        M = cv2.moments(contour)
        if M['m00'] == 0:
            cx, cy = contour.mean(axis=0)[0]  # fallback to mean if zero area
        else:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

        # Scale the contour towards the centroid
        center = np.array([[cx, cy]])
        scaled_contour = center + scale_percent * (contour.reshape(-1, 2) - center)
        scaled_contour = scaled_contour.astype(np.int32).reshape(-1, 1, 2)

        # Draw the scaled contour
        _ = cv2.drawContours(b_mask, [scaled_contour], -1, (255, 255, 255), cv2.FILLED)

        # Choose one:

        # OPTION-1: Isolate object with black background
        mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
        isolated = cv2.bitwise_and(mask3ch, img)
#        isolated_depth = cv2.bitwise_iand(mask3ch, depth)
        isolated_depth = cv2.bitwise_and(depth, depth, mask=b_mask)

        # OPTION-2: Isolate object with transparent background (when saved as PNG)
        #isolated = np.dstack([img, b_mask])

        # OPTIONAL: detection crop (from either OPT1 or OPT2)
        #x1, y1, x2, y2 = c.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
        #iso_crop = isolated[y1:y2, x1:x2]
        name = "isolate_" + filename + "_" + str(label) + str(i) + ".jpg"
        cv2.imwrite(name, isolated)
        name = "isolate_" + filename + "_" + str(label) + str(i) + "_depth.png"
        cv2.imwrite(name, isolated_depth)
        i = i + 1

