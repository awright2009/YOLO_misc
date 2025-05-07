import torch
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image
import torchvision.transforms as T
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor
)
import torch.nn.functional as F

# --- Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Point cloud loader
def load_pcd(path):
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else np.ones_like(pts)
    return pts, colors

pts1, colors1 = load_pcd("left.ply")
pts2, colors2 = load_pcd("right.ply")

pts = np.vstack([pts1, pts2])
colors = np.vstack([colors1, colors2])

pts = torch.tensor(pts, dtype=torch.float32, device=device).unsqueeze(0).requires_grad_()
colors = torch.tensor(colors, dtype=torch.float32, device=device).unsqueeze(0)

# --- Image loader
def load_image(path):
    img = Image.open(path).convert("RGB").resize((1920, 1080))
    return T.ToTensor()(img).permute(1, 2, 0).unsqueeze(0).to(device)

target_img1 = load_image("images/left.JPG")
target_img2 = load_image("images/right.JPG")

# --- Save target reference images for visual comparison
def save_target_image(tensor_img, filename):
    img = tensor_img.squeeze(0).detach().cpu().numpy()  # (H, W, 3)
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(filename)

os.makedirs("rendered_output", exist_ok=True)
save_target_image(target_img1, "rendered_output/target_view_1.png")
save_target_image(target_img2, "rendered_output/target_view_2.png")

# --- View matrix to (R, T) converter
def opengl_view_to_RT(view):
    view = view.copy()
    view[:3, 2] *= -1  # Flip Z axis (OpenGL looks -Z, PyTorch3D expects +Z)
    
    R = view[:3, :3].T  # Transpose rotation
    T = -R @ view[:3, 3]
    return torch.tensor(R, dtype=torch.float32).unsqueeze(0).to(device), torch.tensor(T, dtype=torch.float32).unsqueeze(0).to(device)



# Example: identity views (replace with actual views if needed)
view_gl_1 = np.eye(4, dtype=np.float32)
view_gl_2 = np.eye(4, dtype=np.float32)
R1, T1 = opengl_view_to_RT(view_gl_1)
R2, T2 = opengl_view_to_RT(view_gl_2)

# --- OpenGL-matching intrinsics
width, height = 1920, 1080
fx, fy = width, height
cx, cy = height / 2.0, width / 2.0

# --- Renderer factory
def make_renderer(R, T):
    cameras = PerspectiveCameras(
        device=device,
        R=R,
        T=T,
        in_ndc=False,
        focal_length=((fx, fy),),
        principal_point=((cx, cy),),
        image_size=((height, width),)
    )
    raster_settings = PointsRasterizationSettings(
        image_size=(height, width),
        radius=0.002,  # Adjust to match OpenGL point size
        points_per_pixel=10
    )
    return PointsRenderer(
        rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
        compositor=AlphaCompositor()
    )

renderer1 = make_renderer(R1, T1)
renderer2 = make_renderer(R2, T2)

# --- Point cloud wrapper
def make_pointcloud():
    return Pointclouds(points=[pts.squeeze(0)], features=[colors.squeeze(0)])

# --- Optimization loop
optimizer = torch.optim.Adam([pts], lr=0.01)

for i in range(500):
    optimizer.zero_grad()

    rendered1 = renderer1(make_pointcloud())
    rendered2 = renderer2(make_pointcloud())

    loss = F.mse_loss(rendered1, target_img1) + F.mse_loss(rendered2, target_img2)
    loss.backward()
    optimizer.step()

    if i % 50 == 0:
        print(f"Iter {i} | Loss: {loss.item():.4f}")
        img1 = rendered1[0].detach().cpu().numpy()
        img2 = rendered2[0].detach().cpu().numpy()

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(img1)
        axs[0].set_title("Rendered View 1")
        axs[1].imshow(img2)
        axs[1].set_title("Rendered View 2")
        for ax in axs:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(f"rendered_output/iter_{i:04d}.png")
        plt.close()

# --- Save final point cloud
optimized_pts = pts.detach().cpu().squeeze(0).numpy()
optimized_colors = colors.detach().cpu().squeeze(0).numpy()
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(optimized_pts)
pcd.colors = o3d.utility.Vector3dVector(optimized_colors)
o3d.io.write_point_cloud("optimized_point_cloud.ply", pcd, write_ascii=False)
print("Saved optimized_point_cloud.ply")
