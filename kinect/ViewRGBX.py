import numpy as np
import matplotlib.pyplot as plt
import cv2

# --- Configuration ---
filename = 'data/color0.raw'  # Change to your actual filename
output_bmp = 'image.bmp'
width = 640              # Update as needed
height = 480             # Update as needed

# --- Read raw RGBX data ---
with open(filename, 'rb') as f:
    raw_data = f.read()

image_data = np.frombuffer(raw_data, dtype=np.uint8)

expected_size = width * height * 4
if len(image_data) != expected_size:
    raise ValueError(f"Expected {expected_size} bytes, got {len(image_data)} bytes.")

image_data = image_data.reshape((height, width, 4))

# --- Save to BMP ---
cv2.imwrite(output_bmp, image_data)
print(f"BMP saved to {output_bmp}")

# Extract RGB and convert to BGR
image_bgr = image_data[:, :, :3][:, :, ::-1]



# --- Display the image ---
plt.imshow(image_bgr)
plt.axis('off')
plt.title("RGBX Image (BGR View)")
plt.show()