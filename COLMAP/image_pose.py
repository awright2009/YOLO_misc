import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
import os

def quaternion_to_rotation_matrix(q):
    """
    Converts a quaternion [w, x, y, z] into a 3x3 rotation matrix.
    
    Parameters:
        q (list or array): Quaternion in the form [w, x, y, z]
    
    Returns:
        numpy.ndarray: 3x3 rotation matrix
    """
    w, x, y, z = q

    # Normalize the quaternion to avoid scaling issues
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
    ])
    
    return R



def load_colmap_poses(images_txt_path, invert_poses=False):
    poses_by_name = {}
    
    with open(images_txt_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('#') or line == '':
            i += 1
            continue

        parts = line.split()
        if len(parts) >= 9:
            image_id = int(parts[0])
            qvec = list(map(float, parts[1:5]))  # qw, qx, qy, qz
            tvec = list(map(float, parts[5:8]))  # tx, ty, tz
            image_name = parts[-1]


            poses_by_name[image_name] = {"pos" : tvec, "orientation" : qvec}


            i += 2  # Skip the corresponding 2D-3D point line
        else:
            i += 1

    return poses_by_name

poses = load_colmap_poses("images.txt", invert_poses=True)

if len(sys.argv) < 2:
    print("Usage: python image_pose.py <image_name>")
    print("Note: expects images.txt in same directory as script")
    quit()

filename = os.path.basename(sys.argv[1])
pose = poses[filename]

print("position:\n", pose["pos"])
print("orientation quat:\n", pose["orientation"])
R = quaternion_to_rotation_matrix(pose["orientation"])         
print("orientation matrix:\n", R)


print(pose)
