import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Parameters
num_triangles = 20
lr = 1e-1
steps = 500
image_path = 'images/left.JPG'  # <-- Replace with your image path

# Load and preprocess target RGB image
def load_target_image(path):
    img = Image.open(path).convert('RGB')
    img = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3]
    return torch.tensor(img), img.shape[0], img.shape[1]

target, image_h, image_w = load_target_image(image_path)

# Init triangle vertices and RGB colors
vertices = torch.rand((num_triangles, 3, 2), requires_grad=True)  # Normalized (0..1) coordinates
colors = torch.rand((num_triangles, 3), requires_grad=True)       # RGB colors for each triangle

# Differentiable rasterizer (now supports color)
def rasterize_triangles(verts, colors, height, width):
    img = torch.zeros((height, width, 3), device=verts.device)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, height, device=verts.device),
                            torch.linspace(0, 1, width, device=verts.device), indexing='ij')
    grid = torch.stack([xx, yy], dim=-1)  # [H, W, 2]

    for i, tri in enumerate(verts):
        v0, v1, v2 = tri[0], tri[1], tri[2]

        def edge_func(a, b, p):
            return (p[..., 0] - a[0]) * (b[1] - a[1]) - (p[..., 1] - a[1]) * (b[0] - a[0])

        w0 = edge_func(v1, v2, grid)
        w1 = edge_func(v2, v0, grid)
        w2 = edge_func(v0, v1, grid)

        mask = torch.sigmoid(100 * w0) * torch.sigmoid(100 * w1) * torch.sigmoid(100 * w2)
        mask = mask.unsqueeze(-1)  # [H, W, 1]

        color = colors[i].view(1, 1, 3)  # [1, 1, 3]
        img = torch.clamp(img + mask * color, 0, 1)

    return img

# Optimizer
optimizer = torch.optim.Adam([vertices, colors], lr=lr)

# Setup live plot
plt.ion()
fig, ax = plt.subplots()

# Training loop
for step in range(steps):
    optimizer.zero_grad()
    rendered = rasterize_triangles(vertices, colors, image_h, image_w)
    loss = F.mse_loss(rendered, target.to(rendered.device))
    loss.backward()
    optimizer.step()

    if step % 25 == 0 or step == steps - 1:
        ax.clear()
        ax.imshow(rendered.detach().cpu().numpy())
        ax.set_title(f"Step {step} - Loss: {loss.item():.4f}")
        ax.axis('off')
        plt.pause(0.01)

plt.ioff()
plt.show()
