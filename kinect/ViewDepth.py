import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# --- Configuration ---
filename = 'data/depth0.raw'  # Path to your 16-bit raw file
output_png = 'depth.png'
width = 640             # Set image width
height = 480            # Set image height

# --- Read raw 16-bit depth data ---
with open(filename, 'rb') as f:
    raw_data = f.read()

# Convert to uint16 array
depth_image = np.frombuffer(raw_data, dtype=np.uint16)

expected_size = width * height
if len(depth_image) != expected_size:
    raise ValueError(f"Expected {expected_size} pixels, got {len(depth_image)}")

# Reshape to (height, width)
depth_image = depth_image.reshape((height, width))


# --- Save as 16-bit PNG ---
imageio.imwrite(output_png, depth_image)

# --- Display as grayscale ---
plt.imshow(depth_image, cmap='gray', vmin=np.min(depth_image), vmax=np.max(depth_image))
plt.colorbar(label="Depth Value")
plt.title("16-bit Depth Image")
plt.axis('off')
plt.show()