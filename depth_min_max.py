import cv2
import numpy as np



def load_depth_image(path, max_depth=5.0):
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

image = load_depth_image("images/isolate_right_cup0_depth.png", max_depth=15)

if image is None:
    raise ValueError("Image not loaded. Check the file path.")



min_val = 65535  # max value for 16-bit unsigned int
max_val = 0

height, width = image.shape


# Flatten the image to 1D for efficient computation
non_zero_pixels = image[image != 0]

found_non_zero = non_zero_pixels.size > 0

if found_non_zero:
    min_val = np.min(non_zero_pixels)
else:
    min_val = None  # or some default

max_val = np.max(image)

# If no non-zero values were found
if not found_non_zero:
    min_val = 0


# Output to a text file
with open('output_values.txt', 'w') as f:
    f.write(f"Min (non-zero): {min_val}\n")
    f.write(f"Max: {max_val}\n")

print(f"Min (non-zero): {min_val}, Max: {max_val}")