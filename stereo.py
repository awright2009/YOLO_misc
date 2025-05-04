import numpy as np
import cv2
from matplotlib import pyplot as plt


def ShowDisparity(bSize=5):
    stereo = cv2.StereoBM_create(numDisparities=160, blockSize=bSize)
    disparity = stereo.compute(img_left, img_right)

    # Normalize disparity to the range 0-255
    disparity = cv2.normalize(disparity, disparity, 0, 255, cv2.NORM_MINMAX)
    disparity = np.uint8(disparity)
    
    return disparity

# Load images in grayscale
img_left = cv2.imread('images/left_straight.JPG', cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread('images/right_straight.JPG', cv2.IMREAD_GRAYSCALE)

# Resize images (optional step, adjust size to fit your needs)
new_width = 500  
scale_factor = new_width / img_left.shape[1]
new_height = int(img_left.shape[0] * scale_factor)

img_left = cv2.resize(img_left, (new_width, new_height))
img_right = cv2.resize(img_right, (new_width, new_height))



# Call ShowDisparity with resized images
result = ShowDisparity(bSize=21)
plt.imshow(result, cmap='gray')
plt.axis('off')
plt.show()



# Assuming you have the following known parameters:
focal_length = 1000  # in pixels (example value, you need to measure/calibrate this)
baseline = 0.1  # in meters (the distance between the two cameras)

def disparity_to_depth(disparity_map, focal_length, baseline):
    # Avoid division by zero by masking invalid disparity values (e.g., zero or negative)
    disparity_map = np.float32(disparity_map)
    disparity_map[disparity_map <= 0] = 0.1  # Small constant to avoid division by zero
    
    depth_map = (focal_length * baseline) / disparity_map
    return depth_map

# Example of converting disparity to depth
depth_map = disparity_to_depth(result, focal_length, baseline)

# Display the disparity map in grayscale
plt.imshow(depth_map, cmap='gray')  # 'gray' ensures grayscale display
plt.axis('off')
plt.title("Disparity Map")
plt.show()