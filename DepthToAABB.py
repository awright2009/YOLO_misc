import cv2
import numpy as np
import sys
import random



def load_depth_image(path, max_depth=1.0):
    """
    Loads a depth image and scales it to real-world depth in meters.
    Supports both 8-bit (0–255) and 16-bit (0–65535) grayscale images.
    """
    depth_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if depth_img is None:
        raise FileNotFoundError(f"Could not read: {path}")

    if depth_img.dtype == np.uint8:
        # 8-bit depth: assume 0–255 → 0 to max_depth
        print("Loaded 8-bit depth image.")
        return (depth_img.astype(np.float32) / 255.0) * max_depth

    elif depth_img.dtype == np.uint16:
        # 16-bit depth: assume 0–65535 → 0 to max_depth
        print("Loaded 16-bit depth image.")
        return (depth_img.astype(np.float32) / 65535.0) * max_depth

    else:
        raise ValueError(f"Unsupported depth image format: {depth_img.dtype}")

#image = load_depth_image("images/isolate_right_cup0_depth.png", max_depth=15)
image = load_depth_image(sys.argv[1], max_depth=1)


if image is None:
    raise ValueError("Image not loaded. Check the file path.")



min_val = 65535  # max value for 16-bit unsigned int
max_val = 0

height, width = image.shape


r = random.uniform(0, 1)
g = random.uniform(0, 1)
b = random.uniform(0, 1)


# Flatten the image to 1D for efficient computation
non_zero_pixels = image[image != 0]

found_non_zero = non_zero_pixels.size > 0

if found_non_zero:
    min_val = np.min(non_zero_pixels)
else:
    min_val = None  # or some default


non_zero_coords = np.argwhere(image != 0)

if non_zero_coords.size > 0:
    min_y, min_x = non_zero_coords.min(axis=0)
    max_y, max_x = non_zero_coords.max(axis=0)
else:
    min_x = None
    min_y = None

max_val = np.max(image)

# If no non-zero values were found
if not found_non_zero:
    min_val = 0


# Output to a text file
filename = f"object_depths.txt"
with open(filename, 'a') as f:
    f.write(f"{sys.argv[1]} [{min_x} {min_y} {min_val}, {max_x} {max_y} {max_val}] [{r} {g} {b}]\n")



print(f"{sys.argv[1]}")
print(f"Min Depth: {min_val}, Max: {max_val}")
print(f"Min X: {min_x}, Max X: {max_x}")
print(f"Min Y: {min_y}, Max Y: {max_y}")
print(f"[{min_x} {min_y} {min_val}, {max_x} {max_y} {max_val}] [{r} {g} {b}]\n")