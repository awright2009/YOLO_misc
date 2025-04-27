import numpy as np
import cv2
from PIL import Image
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import glfw

import sys
import numpy


last_x, last_y = 0, 0
yaw, pitch = 0.0, 0.0
left_mouse_pressed = False

cam_offset = np.array([0.0, 0.0, -1.0], dtype=np.float32)  # X, Y, Z

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


def depth_to_mesh(depth, intrinsics, rgb_img=None):
    h, w = depth.shape[:2]
    
    fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']

    i, j = np.indices((h, w), dtype=np.int16)
    
    # prevent divide by zero, zero means far away
    depth[depth == 0] = 1
    z = -1 / depth

   	
    x = (j - cx) * z / fx
    y = (i - cy) * z / fy
    vertices = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    if rgb_img is not None:
        colors = rgb_img.reshape(-1, 3).astype(np.float32) / 255.0
    else:
        gray = (depth / np.max(depth)).clip(0, 1)
        colors = np.stack([gray, gray, gray], axis=-1).reshape(-1, 3).astype(np.float32)

    triangles = np.array([], dtype=np.uint32).reshape(0, 3)
    return vertices.astype(np.float32), colors, triangles

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

def render_mesh_to_image(vertices, colors, indices, width, height, output_path):
    window = init_opengl_context(width, height)

    vertex_shader = """
    #version 330 core
    layout(location = 0) in vec3 aPos;
    layout(location = 1) in vec3 aColor;

    uniform mat4 projection;
    uniform mat4 view;

    out vec3 vColor;

    void main() {
        gl_Position = projection * view * vec4(aPos, 1.0);
        vColor = aColor;
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

    glPointSize(2.0)  # or larger, depending on resolution

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo = glGenBuffers(1)
    vertex_data = np.hstack([vertices, colors])
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)

    #ebo = glGenBuffers(1)
    #glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    #glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    # Layout
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))  # position
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))  # color
    glEnableVertexAttribArray(1)

    # Offscreen buffer
    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)

    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)

    if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
        raise Exception("Framebuffer not complete")

    # Upload projection/view matrices
    proj = perspective_from_intrinsics_infinite_depth(width, height, width / 2.0, height / 2.0, width, height, 250, 250)
    view = identity_view()



    # Draw
    glViewport(0, 0, width, height)
    glClearColor(1, 1, 1, 1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glUseProgram(shader)
    glBindVertexArray(vao)

    proj_loc = glGetUniformLocation(shader, "projection")
    view_loc = glGetUniformLocation(shader, "view")

    glUniformMatrix4fv(proj_loc, 1, GL_TRUE, proj)
    glUniformMatrix4fv(view_loc, 1, GL_TRUE, view)


    glDrawArrays(GL_POINTS, 0, len(vertices))

    # Read pixels
    data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    image = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
    image = np.flip(image, axis=0)  # flip vertically
    image = np.flip(image, axis=1)  # flip horizontally
    Image.fromarray(image).save(output_path)
    print(f"Saved: {output_path}")

    glfw.terminate()


def identity_view():
    return np.eye(4, dtype=np.float32)


def orthographic_projection(left, right, bottom, top, near, far):
    proj = np.zeros((4, 4), dtype=np.float32)

    proj[0, 0] = 2.0 / (right - left)
    proj[1, 1] = 2.0 / (top - bottom)
    proj[2, 2] = -2.0 / (far - near)
    proj[0, 3] = -(right + left) / (right - left)
    proj[1, 3] = -(top + bottom) / (top - bottom)
    proj[2, 3] = -(far + near) / (far - near)
    proj[3, 3] = 1.0

    return proj


def orthographic_projection_infinite_depth(left, right, bottom, top, near):
    proj = np.zeros((4, 4), dtype=np.float32)

    proj[0, 0] = 2.0 / (right - left)  # Scale X
    proj[1, 1] = 2.0 / (top - bottom)  # Scale Y
    proj[2, 2] = -1.0  # Use -1 for infinite depth, no scaling on Z axis
    proj[0, 3] = -(right + left) / (right - left)  # Translate X
    proj[1, 3] = -(top + bottom) / (top - bottom)  # Translate Y
    proj[2, 3] = -(2.0 * near)  # Adjust Z to account for near plane
    proj[3, 3] = 1.0  # Homogeneous coordinate

    return proj


def orbit_view_matrix(yaw, pitch):
    # Identity matrix
    view = np.eye(4, dtype=np.float32)

    # Apply yaw (rotation around Y-axis) - Rotate around vertical axis (up)
    yaw_matrix = np.array([
        [np.cos(yaw), 0, np.sin(yaw), 0],
        [0, 1, 0, 0],
        [-np.sin(yaw), 0, np.cos(yaw), 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    # Apply pitch (rotation around X-axis) - Rotate around horizontal axis
    pitch_matrix = np.array([
        [1, 0, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch), 0],
        [0, np.sin(pitch), np.cos(pitch), 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    # Combine yaw and pitch rotations
    view = np.dot(view, yaw_matrix)
    view = np.dot(view, pitch_matrix)

    return view
    
def orbit_view_matrix_pan(yaw_deg, pitch_deg, radius=1.0, offset=(0.0, 0.0, 0.0)):
    yaw_rad = np.radians(yaw_deg)
    pitch_rad = np.radians(pitch_deg)
    x = radius * np.cos(pitch_rad) * np.sin(yaw_rad)
    y = radius * np.sin(pitch_rad)
    z = radius * np.cos(pitch_rad) * np.cos(yaw_rad)

    eye = np.array([x + offset[0], y + offset[1], z + offset[2]], dtype=np.float32)
    center = np.array([offset[0], offset[1], offset[2]], dtype=np.float32)  # pan applied here
    up = np.array([0, 1, 0], dtype=np.float32)

    # lookAt matrix
    f = center - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)

    view = np.eye(4, dtype=np.float32)
    view[0, :3] = s
    view[1, :3] = u
    view[2, :3] = -f
    view[:3, 3] = eye @ np.array([s, u, -f])

    #print(view)
    return view
    

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
        yaw += dx * 0.05
        pitch += dy * 0.05
        pitch = max(min(pitch, 89.0), -89.0)  # clamp to avoid gimbal lock
    last_x, last_y = xpos, ypos

def key_callback(window, key, scancode, action, mods):
    global cam_offset
    if action in [glfw.PRESS, glfw.REPEAT]:
        if key == glfw.KEY_LEFT:
            cam_offset[0] -= 0.05  # pan left
        elif key == glfw.KEY_RIGHT:
            cam_offset[0] += 0.05  # pan right
        elif key == glfw.KEY_UP:
            cam_offset[2] += 0.05  # pan forward (Z+)
        elif key == glfw.KEY_DOWN:
            cam_offset[2] -= 0.05  # pan backward (Z-)
        elif key == glfw.KEY_ENTER:
            cam_offset[1] -= 0.05  # pan up (Y+)
        elif key == glfw.KEY_LEFT_SHIFT or key == glfw.KEY_RIGHT_SHIFT:
            cam_offset[1] += 0.05  # pan down (Y-)

def render_point_cloud_live(vertices, colors, width, height):
    window = init_opengl_context(width, height)
    
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_position_callback)
    glfw.set_key_callback(window, key_callback)

    vertex_shader = """
    #version 330 core
    layout(location = 0) in vec3 aPos;
    layout(location = 1) in vec3 aColor;

    uniform mat4 projection;
    uniform mat4 view;

    out vec3 vColor;

    void main() {
        gl_Position = projection * view * vec4(aPos, 1.0);
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

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo = glGenBuffers(1)
    vertex_data = np.hstack([vertices, colors])
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))  # position
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))  # color
    glEnableVertexAttribArray(1)

    
    
    # Define the orthographic projection with infinite depth
    left = -2
    right = 2
    bottom = -2
    top = 2
    near = 0.001  # Set near plane

    # Use the modified infinite depth orthographic projection
    #proj = orthographic_projection_infinite_depth(left, right, bottom, top, near)
    proj = perspective_from_intrinsics_infinite_depth(width, height, width / 2.0, height / 2.0, width, height, -250, 250)
  
    
    #view = identity_view()
    view = orbit_view_matrix_pan(yaw, pitch, offset=cam_offset)

    proj_loc = glGetUniformLocation(shader, "projection")
    view_loc = glGetUniformLocation(shader, "view")

    glUseProgram(shader)
    glUniformMatrix4fv(proj_loc, 1, GL_TRUE, proj)
    glUniformMatrix4fv(view_loc, 1, GL_TRUE, view)

    glEnable(GL_PROGRAM_POINT_SIZE)
    glEnable(GL_DEPTH_TEST)

    while not glfw.window_should_close(window):
        view = orbit_view_matrix_pan(yaw, pitch, offset=cam_offset)
        #print(cam_offset)
        glUniformMatrix4fv(view_loc, 1, GL_TRUE, view)
        glfw.poll_events()
        glClearColor(0, 0, 0, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glBindVertexArray(vao)
        glDrawArrays(GL_POINTS, 0, len(vertices))

        glfw.swap_buffers(window)

    glfw.terminate()

def perspective_from_intrinsics_infinite_depth(fx, fy, cx, cy, width, height, shift_x, shift_y, near=0.1):
    proj = np.zeros((4, 4), dtype=np.float32)

    cx += shift_x
    cy += shift_y

    proj[0, 0] = -2 * fx / width
    proj[1, 1] = 2 * fy / height
    proj[0, 2] = 1 - 2 * cx / width
    proj[1, 2] = 2 * cy / height - 1
    proj[2, 2] = -1  # limit of -(far + near)/(far - near) as far -> ∞
    proj[2, 3] = -2 * near  # limit of -(2 * far * near)/(far - near) as far -> ∞
    proj[3, 2] = -1
    return proj



def save_mesh_to_obj(filename, vertices, indices, colors=None):
    """
    Saves a triangle mesh to a Wavefront OBJ file.
    Supports vertex colors via MeshLab-style comments (non-standard).
    """
    with open(filename, 'w') as f:
        for i, v in enumerate(vertices):
            if colors is not None:
                c = colors[i]
                f.write(f"v {v[0]} {v[1]} {v[2]} {c[0]} {c[1]} {c[2]}\n")
            else:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        for tri in indices:
            # OBJ uses 1-based indexing
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
    print(f"Saved OBJ to {filename}")

# Run it all together
if __name__ == "__main__":
    numpy.set_printoptions(threshold=sys.maxsize)
    depth_img = load_depth_image("images/left_small.png", max_depth=1)
    
    
    #rgb_img = cv2.cvtColor(cv2.imread("images/left.JPG"), cv2.COLOR_BGR2RGB)
    
    rgb_fullres = cv2.cvtColor(cv2.imread("images/left.JPG"), cv2.COLOR_BGR2RGB)
    # Sample every other pixel to downscale by 2x
    rgb_img = rgb_fullres[::2, ::2]
    

    height, width = depth_img.shape[:2]

    intrinsics = {'fx': width, 'fy': height, 'cx': height / 2.0, 'cy': width / 2.0}

    verts, colors, tris = depth_to_mesh(depth_img, intrinsics, rgb_img)

    #render_mesh_to_image(verts, colors, tris, width, height, "images/left_small_normal.png")
    render_point_cloud_live(verts, colors, width, height)
    save_mesh_to_obj("output_mesh.obj", verts, tris, colors)
