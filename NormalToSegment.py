import numpy as np
from scipy.ndimage import label
from PIL import Image
import sys

def group_normals(normal_map, angle_tolerance_degrees=5, connectivity=4):
    """
    Groups similar normals in the normal_map that are both directionally similar
    and spatially connected. Returns a labeled image where each region has a unique label.
    """
    H, W, _ = normal_map.shape
    normals = normal_map.reshape(-1, 3)
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)  # normalize

    # Step 1: Assign unique ID to each pixel
    unique_ids = np.arange(H * W).reshape(H, W)

    # Step 2: Precompute cosine threshold
    cos_thresh = np.cos(np.radians(angle_tolerance_degrees))

    # Step 3: Build mask of similar-normal neighbors
    mask = np.ones((H, W), dtype=bool)

    def similar(p1, p2):
        return np.dot(p1, p2) > cos_thresh

    # Prepare the output label map
    labels = -np.ones((H, W), dtype=np.int32)
    current_label = 0

    visited = np.zeros((H, W), dtype=bool)
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)] if connectivity == 4 else [(-1, -1), (-1, 0), (-1, 1),
                                                                         (0, -1),           (0, 1),
                                                                         (1, -1), (1, 0),  (1, 1)]

    from collections import deque

    # Step 4: Flood fill with normal similarity
    for y in range(H):
        for x in range(W):
            if visited[y, x]:
                continue

            queue = deque()
            queue.append((y, x))
            visited[y, x] = True
            labels[y, x] = current_label
            n0 = normal_map[y, x] / (np.linalg.norm(normal_map[y, x]) + 1e-8)

            while queue:
                cy, cx = queue.popleft()
                for dy, dx in dirs:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx]:
                        n1 = normal_map[ny, nx]
                        n1 = n1 / (np.linalg.norm(n1) + 1e-8)
                        if np.dot(n0, n1) > cos_thresh:
                            labels[ny, nx] = current_label
                            visited[ny, nx] = True
                            queue.append((ny, nx))

            current_label += 1

    return labels






def labels_to_color_image(labels, seed=42):
    """
    Converts a label map to an RGB image with random colors for each label.
    """
    np.random.seed(seed)  # For reproducibility

    unique_labels = np.unique(labels)
    num_labels = unique_labels.shape[0]

    # Generate a random color for each label (R, G, B in [0, 255])
    colors = np.random.randint(0, 256, size=(num_labels, 3), dtype=np.uint8)

    # Create a mapping from label to color
    label_to_color = {label: color for label, color in zip(unique_labels, colors)}

    # Build the color image
    H, W = labels.shape
    color_image = np.zeros((H, W, 3), dtype=np.uint8)

    for label, color in label_to_color.items():
        color_image[labels == label] = color

    return color_image


if len(sys.argv) < 3:
    print("Usage NormalToSegment.py <file> <tolerance in degrees>\n")
    quit()

normal_map_img = np.array(Image.open(sys.argv[1])).astype(np.float32) / 255.0
labels = group_normals(normal_map_img, angle_tolerance_degrees=float(sys.argv[2]))

# Generate color image
color_labels_img = labels_to_color_image(labels)

# Save or display
filename = sys.argv[1] + "_segmented_labels.png"

Image.fromarray(color_labels_img).save(filename)

# Optional: view inline with matplotlib
#import matplotlib.pyplot as plt
#plt.imshow(color_labels_img)
#plt.axis('off')
#plt.title('Random Colored Label Regions')
#plt.show()

