from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO

m = YOLO("yolo11n-seg.pt")  


res = m.predict()  



# Iterate detection results 


for r in res:
    img = np.copy(r.orig_img)
    img_name = Path(r.path).stem

    # Iterate each object contour 


    for ci, c in enumerate(r):
        label = c.names[c.boxes.cls.tolist().pop()]

        b_mask = np.zeros(img.shape[:2], np.uint8)

        # Create contour mask 


        contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
        _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

        # Choose one:

        # OPTION-1: Isolate object with black background
        mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
        isolated = cv2.bitwise_and(mask3ch, img)

        # OPTION-2: Isolate object with transparent background (when saved as PNG)
        #isolated = np.dstack([img, b_mask])

        # OPTIONAL: detection crop (from either OPT1 or OPT2)
        #x1, y1, x2, y2 = c.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
        #iso_crop = isolated[y1:y2, x1:x2]



