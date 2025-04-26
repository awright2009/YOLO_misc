import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from PIL import Image

def set_infinite_perspective(fov_y_deg, aspect, z_near):
    f = 1.0 / np.tan(np.radians(fov_y_deg) / 2.0)
    proj = np.array([
        [f / aspect, 0,  0,              0],
        [0,          f,  0,              0],
        [0,          0, -1,             -2 * z_near],
        [0,          0, -1,              0]
    ], dtype=np.float32)
    glLoadMatrixf(proj.T)  # OpenGL expects column-major order

def load_obj_vertices(filepath):
    vertices = []
    faces = []

    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

            elif line.startswith('f '):
                parts = line.strip().split()[1:]
                face = [int(p.split('/')[0]) - 1 for p in parts]  # Convert to 0-based

                # Triangulate if more than 3 verts
                for i in range(1, len(face) - 1):
                    faces.append([face[0], face[i], face[i + 1]])

    return np.array(vertices, dtype=np.float32), faces

def init_gl_context(width, height):
    if not glfw.init():
        raise RuntimeError("GLFW init failed")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(width, height, "Offscreen", None, None)
    glfw.make_context_current(window)
    return window

def render_depth_to_png(obj_path, output_path, width=512, height=512, near=0.1, far=5.0):
    window = init_gl_context(width, height)
    glViewport(0, 0, width, height)
    glEnable(GL_DEPTH_TEST)
    glClearColor(1, 1, 1, 1)

    # Set projection


    # In your rendering setup
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    set_infinite_perspective(fov_y_deg=60, aspect=width/height, z_near=0.1)

    # Modelview
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(0, 3, 15.0,   # camera position (eye)
          0, 0, 0,      # look at center
          0, 1, 0)      # up vector
    
    # Rotate the object
    glTranslatef(0.4, 3.0, 0)
    glRotatef(0, 1, 0, 0)
    glRotatef(-20, 0, 1, 0)
    glRotatef(0, 0, 0, 1)

    # Load mesh
    vertices, faces = load_obj_vertices(obj_path)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glBegin(GL_TRIANGLES)
    for face in faces:
        for idx in face:
            glVertex3fv(vertices[idx])
    glEnd()

    # Read and convert depth buffer
    depth_buffer = glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT)
    depth = np.frombuffer(depth_buffer, dtype=np.float32).reshape(height, width)
    depth = np.flip(depth, axis=0)

    # Convert to real depth values
    z_near, z_far = near, far
    depth_linear = (2.0 * z_near * z_far) / (z_far + z_near - (2.0 * depth - 1.0) * (z_far - z_near))
    depth_clipped = np.clip(depth_linear, z_near, z_far)
    depth_16bit = ((depth_clipped - z_near) / (z_far - z_near) * 65535).astype(np.uint16)

    # Save to file
    Image.fromarray(depth_16bit, mode='I;16').save(output_path)
    print(f"Saved depth image: {output_path}")

    glfw.terminate()

# Example usage
render_depth_to_png("depth_scene.obj", "16depth.png")