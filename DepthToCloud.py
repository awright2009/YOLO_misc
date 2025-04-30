import numpy as np
import cv2

from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import glfw

import sys
import math

import re


last_x, last_y = 0, 0
yaw, pitch = 0.0, 0.0
left_mouse_pressed = False


up 			= [0, 1, 0]
forward = [0, 0, 1]

position = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # X, Y, Z

old_x = 0
old_y = 0

data = []

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


def transform_point(point, height, width, intrinsics):
    h = height
    w = width

    fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']

    z = -1 / point[2]

    x = (point[0] - cx) * z / fx
    y = (point[1] - cy) * z / fy

    return [x, y, z]


def transform_points_numpy(depth, intrinsics):
    h, w = depth.shape[:2]

    fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']

    i, j = np.indices((h, w), dtype=np.int16)

    # prevent divide by zero, zero means far away
    depth[depth == 0] = 1
    z = -1 / depth

    x = (j - cx) * z / fx
    y = (i - cy) * z / fy
    vertices = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    return vertices

def depth_to_mesh(depth, intrinsics, rgb_img=None):

    vertices = transform_points_numpy(depth, intrinsics)

    if rgb_img is not None:
        colors = rgb_img.reshape(-1, 3).astype(np.float32) / 255.0
    else:
        gray = (depth / np.max(depth)).clip(0, 1)
        colors = np.stack([gray, gray, gray], axis=-1).reshape(-1, 3).astype(np.float32)

    triangles = np.array([], dtype=np.uint32).reshape(0, 3)
    return vertices.astype(np.float32), colors, triangles


def init_opengl_context(width, height):
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.VISIBLE, glfw.TRUE)  # <--- Make window visible
    window = glfw.create_window(width, height, "Depth Point Cloud Viewer", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")
    glfw.make_context_current(window)
    return window


def identity_view():
    return np.eye(4, dtype=np.float32)


def dot_product(VecA, VecB):
    return VecA[0] * VecB[0] + VecA[1] * VecB[1] + VecA[2] * VecB[2]

def cross_product(VecA, VecB):
    result = [0, 0, 0]
    result[0] = VecA[1] * VecB[2] - VecA[2] * VecB[1]
    result[1] = VecA[2] * VecB[0] - VecA[0] * VecB[2]
    result[2] = VecA[0] * VecB[1] - VecA[1] * VecB[0]
    return result

def normalize(vec):
		mag = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])
		vec[0] = vec[0] / mag
		vec[1] = vec[1] / mag
		vec[2] = vec[2] / mag
		return vec
	
	
def rotate_vector(rad, vec, axis):
    m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    sinVal = math.sin(rad)
    cosVal = math.cos(rad)
    minusVal = 1.0 - cosVal

    m[0][0] = cosVal + minusVal * axis[0] * axis[0];
    m[0][1] = minusVal * axis[0] * axis[1] - sinVal * axis[2];
    m[0][2] = minusVal * axis[0] * axis[2] + sinVal * axis[1];

    m[1][0] = minusVal * axis[1] * axis[0] + sinVal * axis[2];
    m[1][1] = cosVal + minusVal * axis[1] * axis[1];
    m[1][2] = minusVal * axis[1] * axis[2] - sinVal * axis[0];

    m[2][0] = minusVal * axis[2] * axis[0] - sinVal * axis[1];
    m[2][1] = minusVal * axis[2] * axis[1] + sinVal * axis[0];
    m[2][2] = cosVal + minusVal * axis[2] * axis[2];


    result = [1, 2, 3]
    #result = np.matmul(vec, m)
    # matrix mult
    result[0] = vec[0] * m[0][0] + vec[1] * m[1][0] + vec[2] * m[2][0]
    result[1] = vec[0] * m[0][1] + vec[1] * m[1][1] + vec[2] * m[2][1]
    result[2] = vec[0] * m[0][2] + vec[1] * m[1][2] + vec[2] * m[2][2]
		
    return result

    
def view_matrix(yaw_deg, pitch_deg,  pos):
    global forward
    global up

    yaw_rad = np.radians(yaw_deg)
    pitch_rad = np.radians(pitch_deg)
    scale = 50.0
    #up = np.array([0, 1, 0], dtype=np.float32)
    vup = [0, 1, 0]
    
    
    # Left / Right
    forward = rotate_vector(scale * (yaw_rad / 100.0), forward, vup )
    up = rotate_vector(scale * (yaw_rad / 100.0), up, vup)

    # Up / Down
    right = cross_product(up, forward)
    right = normalize(right)

    old_forward = forward
    old_up = up

    forward = rotate_vector(scale * (pitch_rad / 100.0), forward, right)
    up = rotate_vector(scale * (pitch_rad / 100.0), up, right)
    forward = normalize(forward)
    up = normalize(up)

    if (up[0] * vup[0] + up[1] * vup[1] + up[2] * vup[2]  < 0.004):
        forward = old_forward
        up = old_up

    
    view = np.eye(4, dtype=np.float32)
    view[0][0] = right[0]
    view[1][0] = up[0]
    view[2][0] = forward[0]

    view[0][1] = right[1]
    view[1][1] = up[1]
    view[2][1] = forward[1]

    view[0][2] = right[2]
    view[1][2] = up[2]
    view[2][2] = forward[2]

    view[0][3] = dot_product(right, pos)
    view[1][3] = dot_product(up, pos)
    view[2][3] = dot_product(forward, pos)
    #print(view)
    return view
    
def perspective(fovy, aspect, z_near, z_far, infinite=False):
    radians = math.radians(fovy / 2)
    cotangent = math.cos(radians) / math.sin(radians)
    delta_z = z_far - z_near
    epsilon = 0.001

    m = np.zeros((4, 4), dtype=np.float32)

    m[0, 0] = cotangent / aspect
    m[1, 1] = cotangent
    m[2, 2] = -(z_far + z_near) / delta_z
    m[2, 3] = -1.0
    m[3, 2] = -2 * z_near * z_far / delta_z

    if infinite:
        m[2, 2] = epsilon - 1.0
        m[3, 2] = z_near * (epsilon - 2.0)

    return m

def mouse_button_callback(window, button, action, mods):
    global left_mouse_pressed
    if button == glfw.MOUSE_BUTTON_LEFT:
        if action == glfw.PRESS:
            left_mouse_pressed = True
        elif action == glfw.RELEASE:
            left_mouse_pressed = False

def cursor_position_callback(window, xpos, ypos):
    global last_x, last_y, yaw, pitch, left_mouse_pressed
    if left_mouse_pressed:
        dx = xpos - last_x
        dy = ypos - last_y
        yaw -= dx * 0.05
        pitch += dy * 0.05
    last_x, last_y = xpos, ypos

def key_callback(window, key, scancode, action, mods):
    global position
    global up
    global forward

    right = cross_product(up, forward)

    scale = 0.05
    if action in [glfw.PRESS, glfw.REPEAT]:
        if key == glfw.KEY_LEFT:
            position -= [item * scale for item in right]
        elif key == glfw.KEY_RIGHT:
            position += [item * scale for item in right]
        elif key == glfw.KEY_ENTER:
            position -= [item * scale for item in up]
        elif key == glfw.KEY_LEFT_SHIFT or key == glfw.KEY_RIGHT_SHIFT:
            position += [item * scale for item in up]
        elif key == glfw.KEY_UP:
            position += [item * scale for item in forward]
        elif key == glfw.KEY_DOWN:
            position -= [item * scale for item in forward]


    
    
def triangles_from_aabb_with_color(aabb, color):
    cube_triangles = [
        [0, 4, 5], [0, 5, 1],
        [2, 3, 7], [2, 7, 6],
        [0, 1, 3], [0, 3, 2],
        [4, 6, 7], [4, 7, 5],
        [0, 2, 6], [0, 6, 4],
        [1, 5, 7], [1, 7, 3]
    ]

    triangle_vertices = []

    for tri in cube_triangles:
        for idx in tri:
            vertex = aabb[idx]
            triangle_vertices.extend([vertex[0], vertex[1], vertex[2], color[0], color[1], color[2] ] )
    return triangle_vertices
    


def make_aabb(min_xyz, max_xyz):
    min_x, min_y, min_z = min_xyz
    max_x, max_y, max_z = max_xyz

    aabb = [
        [min_x, min_y, min_z],  # i = 0b000
        [min_x, min_y, max_z],  # i = 0b001
        [min_x, max_y, min_z],  # i = 0b010
        [min_x, max_y, max_z],  # i = 0b011
        [max_x, min_y, min_z],  # i = 0b100
        [max_x, min_y, max_z],  # i = 0b101
        [max_x, max_y, min_z],  # i = 0b110
        [max_x, max_y, max_z],  # i = 0b111
    ]
        
    return aabb

def generate_aabb(data_items, height, width, intrinsics):
    all_triangles = []
    for item in data_items:
        min_point = transform_point(item['min_xyz'], height, width, intrinsics)
        max_point = transform_point(item['max_xyz'], height, width, intrinsics)
        aabb = make_aabb(min_point, max_point)
        tri_data = triangles_from_aabb_with_color(aabb, item['color'])
        all_triangles.append(tri_data)
        
    if not all_triangles:
        return np.array([], dtype=np.float32)

    return np.concatenate(all_triangles, axis=0)

def render_point_cloud_live(vertices, colors, width, height):
    global old_x
    global old_y
    window = init_opengl_context(width, height)
    
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_position_callback)
    glfw.set_key_callback(window, key_callback)
    vertex_data = np.hstack([vertices, colors])

    vertex_shader = """
    #version 330 core
    layout(location = 0) in vec3 aPos;
    layout(location = 1) in vec3 aColor;

    uniform mat4 projection;
    uniform mat4 view;

    out vec3 vColor;

    void main() {
        gl_Position = (projection * view)  * vec4(aPos, 1.0);
        vColor = aColor;
        gl_PointSize = 2.0;
    }
    """

    fragment_shader = """
    #version 330 core
    in vec3 vColor;
    out vec4 FragColor;

    void main() {
        FragColor = vec4(vColor, 1.0);
    }
    """

    shader = compileProgram(
        compileShader(vertex_shader, GL_VERTEX_SHADER),
        compileShader(fragment_shader, GL_FRAGMENT_SHADER)
    )

    glDisable(GL_CULL_FACE)

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    glEnableVertexAttribArray(0)
    glEnableVertexAttribArray(1)

    vbo = glGenBuffers(1)

    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))  # position
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))  # color
    glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data.astype(np.float32), GL_STATIC_DRAW)



    box_vertex_data = generate_aabb(data, height, width, intrinsics)
    box_vertex_data = box_vertex_data.astype(np.float32)
    
    box_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, box_vbo)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))  # position
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))  # color
    glBufferData(GL_ARRAY_BUFFER, box_vertex_data.nbytes, box_vertex_data, GL_STATIC_DRAW)


    glBlendFunc(GL_ONE, GL_ONE)
    
    
    # Define the orthographic projection with infinite depth
    left = -2
    right = 2
    bottom = -2
    top = 2
    near = 0.001  # Set near plane

    # Use the modified infinite depth orthographic projection
    #proj = orthographic_projection_infinite_depth(left, right, bottom, top, near)
    proj = perspective_from_intrinsics_infinite_depth(width, height, width / 2.0, height / 2.0, width, height, -250, 250)
    #proj = perspective(fovy=36.87, aspect=2.0, z_near=0.1, z_far=1000.0, infinite=True)

    delta_x = yaw - old_x
    delta_y = (pitch - old_y)

    old_x = yaw
    old_y = pitch
    
    #view = identity_view()
    view = view_matrix(delta_x, delta_y, position)

    proj_loc = glGetUniformLocation(shader, "projection")
    view_loc = glGetUniformLocation(shader, "view")

    glUseProgram(shader)
    glUniformMatrix4fv(proj_loc, 1, GL_TRUE, proj)
    glUniformMatrix4fv(view_loc, 1, GL_TRUE, view)

    glEnable(GL_PROGRAM_POINT_SIZE)
    glEnable(GL_DEPTH_TEST)



    while not glfw.window_should_close(window):
        delta_x = yaw - old_x
        delta_y = pitch - old_y

        old_x = yaw
        old_y = pitch
        view = view_matrix(delta_x, delta_y, position)
        #print(cam_offset)
        glUniformMatrix4fv(view_loc, 1, GL_TRUE, view)
        glfw.poll_events()
        glClearColor(0, 0, 0, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))  # position
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))  # color
        glDrawArrays(GL_POINTS, 0, len(vertices))

        glEnable(GL_BLEND)
        
        glBindBuffer(GL_ARRAY_BUFFER, box_vbo)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))  # position
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))  # color
        glDrawArrays(GL_TRIANGLES, 0, len(box_vertex_data))
        glDisable(GL_BLEND)

        glfw.swap_buffers(window)

    glfw.terminate()
    
    
    
def parse_file(filepath):
    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            # Match using regular expressions
            match = re.match(
                r'(\S+)\s+\[([^\]]+)\]\s+\[([^\]]+)\]',
                line
            )
            if not match:
                print(f"Line skipped (invalid format): {line}")
                continue

            filename = match.group(1)
            bbox_str = match.group(2)
            color_str = match.group(3)

            try:
                # Split bounding box into two 3D points
                bbox_parts = bbox_str.split(',')
                min_xyz = list(map(float, bbox_parts[0].strip().split()))
                max_xyz = list(map(float, bbox_parts[1].strip().split()))

                # Split RGB color
                rgb = list(map(float, color_str.strip().split()))

                data.append({
                    'filename': filename,
                    'min_xyz': min_xyz,
                    'max_xyz': max_xyz,
                    'color': rgb
                })
            except Exception as e:
                print(f"Error parsing line: {line}\n{e}")

    return data


def perspective_from_intrinsics_infinite_depth(fx, fy, cx, cy, width, height, shift_x, shift_y, near=0.1):
    proj = np.zeros((4, 4), dtype=np.float32)

    cx += shift_x
    cy += shift_y

    # diagnol
    proj[0][0] = -2 * fx / width
    proj[1][1] = 2 * fy / height
    proj[2][2] = -1  # limit of -(far + near)/(far - near) as far -> ∞


    proj[0][2] = 1 - 2 * cx / width
    proj[1][2] = 2 * cy / height - 1
    proj[3][2] = -1
    proj[2][3] = -2 * near  # limit of -(2 * far * near)/(far - near) as far -> ∞

    return proj



if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage python DepthToCloud.py <rgb_file> <depth_file> <aabb_file>\n")
        exit()

    print(f"RGB Image {sys.argv[1]} Depth Image {sys.argv[2]}")
    rgb_img = cv2.cvtColor( cv2.imread(sys.argv[1]), cv2.COLOR_BGR2RGB)
    depth_img = load_depth_image(sys.argv[2], max_depth=1)


    if len(sys.argv) == 4:
        data = parse_file(sys.argv[3])
    else:
        data = np.zeros(0)
   

    height, width = depth_img.shape[:2]

    intrinsics = {'fx': width, 'fy': height, 'cx': height / 2.0, 'cy': width / 2.0}

    verts, colors, tris = depth_to_mesh(depth_img, intrinsics, rgb_img)

    render_point_cloud_live(verts, colors, width, height)

