import numpy as np
import cv2
from PIL import Image
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import glfw

import sys
import numpy
from pathlib import Path
import sys



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


def depth_to_mesh(depth, intrinsics):
    h, w = depth.shape[:2]
    
    fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']

    i, j = np.indices((h, w), dtype=np.int16)
    
    # prevent divide by zero, zero means far away
    depth[depth == 0] = 1
    z = -1 / depth


    x = (j - cx) * z / fx
    y = (i - cy) * z / fy
    vertices = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    # Build triangle indices
    triangles = []
    for i in range(h - 1):
        for j in range(w - 1):
            idx = i * w + j
            tri1 = [idx, idx + 1, idx + w]
            tri2 = [idx + 1, idx + w + 1, idx + w]
            triangles.extend([tri1, tri2])

    triangles = np.array(triangles, dtype=np.uint32)
    #print(triangles)

    # Calculate triangle normals
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(face_normals, axis=1, keepdims=True)
    norm[norm == 0] = 1
    face_normals /= norm
    #print(face_normals)

    # Per-vertex normal = average of connected face normals
    vertex_normals = np.zeros_like(vertices)
    counts = np.zeros((vertices.shape[0], 1))
    for i, tri in enumerate(triangles):
        for v in tri:
            vertex_normals[v] += face_normals[i]
            counts[v] += 1
    counts[counts == 0] = 1
    vertex_normals /= counts

    # Normalize normals and map to [0, 1] RGB
    norm = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norm[norm == 0] = 1
    vertex_normals /= norm
    colors = (vertex_normals + 1.0) / 2.0  # [-1,1] → [0,1]
    #print(colors)

    return vertices.astype(np.float32), colors.astype(np.float32), triangles

def init_opengl_context(width, height):
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(width, height, "Offscreen", None, None)
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


    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo = glGenBuffers(1)
    vertex_data = np.hstack([vertices, colors])
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)

    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

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
    proj = perspective_from_intrinsics_infinite_depth(width, height, width / 2.0, height / 2.0, width, height, 500, 500)
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


    glDrawElements(GL_TRIANGLES, len(indices) * 3, GL_UNSIGNED_INT, None)

    # Read pixels
    data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    image = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
    image = np.flip(image, axis=0)  # flip vertically
    image = np.flip(image, axis=1)  # flip horizontally
    Image.fromarray(image).save(output_path)
    print(f"Saved: {output_path}")

    glfw.terminate()

def perspective_from_intrinsics_infinite_depth(fx, fy, cx, cy, width, height, shift_x, shift_y, near=0.1):
    proj = np.zeros((4, 4), dtype=np.float32)

    cx += shift_x
    cy += shift_y

    proj[0, 0] = 2 * fx / width
    proj[1, 1] = 2 * fy / height
    proj[0, 2] = 1 - 2 * cx / width
    proj[1, 2] = 2 * cy / height - 1
    proj[2, 2] = -1  # limit of -(far + near)/(far - near) as far -> ∞
    proj[2, 3] = -2 * near  # limit of -(2 * far * near)/(far - near) as far -> ∞
    proj[3, 2] = -1
    return proj

def identity_view():
    return np.eye(4, dtype=np.float32)

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
    depth_img = load_depth_image(sys.argv[1], max_depth=1)


    height, width = depth_img.shape[:2]

    intrinsics = {'fx': width, 'fy': height, 'cx': height / 2.0, 'cy': width / 2.0}

    verts, colors, tris = depth_to_mesh(depth_img, intrinsics)


    base = Path(sys.argv[1]).stem
    filename = f"{base}_normal.png"

    mesh_name = f"{base}_mesh.obj"

    render_mesh_to_image(verts, colors, tris, width, height, filename)
    print("Generating OBJ\n")
    save_mesh_to_obj(mesh_name, verts, tris, colors)
