import cv2
import numpy as np
from collections import defaultdict
import os
import sys

def find_largest_color_groups(image_path, output_dir='output_groups'):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image at {image_path}")
    h, w, _ = img.shape

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Convert to RGB for consistent color indexing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create a label map to track connected components by color
    color_components = defaultdict(list)

    # For performance, keep a visited map
    visited = np.zeros((h, w), dtype=bool)

    def flood_fill(y, x, color, label_id):
        # Standard 4-connected flood fill
        queue = [(y, x)]
        mask = []
        while queue:
            cy, cx = queue.pop()
            if visited[cy, cx]:
                continue
            if np.all(img_rgb[cy, cx] == color):
                visited[cy, cx] = True
                mask.append((cy, cx))
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                        queue.append((ny, nx))
        if mask:
            color_components[label_id] = (color, mask)

    # Scan each pixel
    label_id = 0
    for y in range(h):
        for x in range(w):
            if not visited[y, x]:
                color = img_rgb[y, x]
                flood_fill(y, x, color, label_id)
                label_id += 1

    # Sort groups by size
    sorted_groups = sorted(color_components.items(), key=lambda item: len(item[1][1]), reverse=True)[:4]

    # Save each as masked image
    for i, (label, (color, coords)) in enumerate(sorted_groups):
        mask_img = np.zeros_like(img_rgb)
        for y, x in coords:
            mask_img[y, x] = color
        output_path = os.path.join(output_dir, f"group_{i+1}.png")
        cv2.imwrite(output_path, cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR))
        print(f"Saved: {output_path} with color {tuple(color)} and {len(coords)} pixels")

# Example usage:
if len(sys.argv) < 2:
   print("Usage: python SegmentToMask.py <segmented_color_image>\n")
   quit()

find_largest_color_groups(sys.argv[1])