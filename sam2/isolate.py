from ultralytics import SAM
from pathlib import Path
import os
import random
import cv2
import numpy as np


filename = "left"

#sam = SAM("sam2_b.pt")
sam = SAM("sam2.1_l.pt")

sam.info()

#model(source=0, show=True, save=True)


results = sam.predict(source="left.JPG", show=False, save=True)

for r in results:
    print(f"Detected {len(r.masks)} masks")
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


        contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
        _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

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

