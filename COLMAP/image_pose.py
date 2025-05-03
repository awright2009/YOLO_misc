import numpy as np
from scipy.spatial.transform import Rotation as R

def colmap_pose_to_matrix(qvec, tvec):
    # Convert quaternion to rotation matrix
    R_wc = R.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()
    t_wc = np.array(tvec).reshape((3, 1))

    # 4x4 transformation matrix (world to camera)
    T_wc = np.eye(4)
    T_wc[:3, :3] = R_wc
    T_wc[:3, 3] = t_wc.flatten()
    
    return T_wc

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

            T_wc = colmap_pose_to_matrix(qvec, tvec)

            if invert_poses:
                T_cw = np.linalg.inv(T_wc)
                poses_by_name[image_name] = T_cw
            else:
                poses_by_name[image_name] = T_wc

            i += 2  # Skip the corresponding 2D-3D point line
        else:
            i += 1

    return poses_by_name

poses = load_colmap_poses("images.txt", invert_poses=True)

# Access the pose of "image001.jpg"
pose = poses["image001.jpg"]
print(pose)
