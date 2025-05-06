import torch
import open3d as o3d
import numpy as np

import matplotlib.pyplot as plt
import os

from PIL import Image
import torchvision.transforms as T
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor
)
import torch.nn.functional as F

# --- Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Helper to load a point cloud
def load_pcd(path):
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points)
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
    else:
        colors = np.ones_like(pts)
    return pts, colors

# --- Load and merge point clouds
pts1, colors1 = load_pcd("left.ply")
pts2, colors2 = load_pcd("right.ply")

pts = np.vstack([pts1, pts2])
colors = np.vstack([colors1, colors2])

pts = torch.tensor(pts, dtype=torch.float32, device=device).unsqueeze(0).requires_grad_()
colors = torch.tensor(colors, dtype=torch.float32, device=device).unsqueeze(0)

# --- Load target reference images
def load_image(path):
    img = Image.open(path).convert("RGB").resize((256, 256))
    return T.ToTensor()(img).permute(1, 2, 0).unsqueeze(0).to(device)  # shape: (1, H, W, 3)

target_img1 = load_image("images/left.JPG")
target_img2 = load_image("images/right.JPG")

# --- Camera extrinsics
R1 = torch.eye(3).unsqueeze(0).to(device)
T1 = torch.zeros(1, 3).to(device)

R2 = torch.tensor([[0., 0., 1.],
                   [0., 1., 0.],
                   [-1., 0., 0.]]).unsqueeze(0).to(device)
T2 = torch.tensor([[0., 0., -1.0]]).to(device)

# --- Point cloud renderer setup
def make_renderer(R, T):
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    raster_settings = PointsRasterizationSettings(
        image_size=256,
        radius=0.01,
        points_per_pixel=10
    )
    return PointsRenderer(
        rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
        compositor=AlphaCompositor()
    )

renderer1 = make_renderer(R1, T1)
renderer2 = make_renderer(R2, T2)

def make_pointcloud():
    return Pointclouds(points=[pts.squeeze(0)], features=[colors.squeeze(0)])

# --- Optimizer
optimizer = torch.optim.Adam([pts], lr=0.01)

os.makedirs("rendered_output", exist_ok=True)

# --- Optimization loop
for i in range(500):
    optimizer.zero_grad()

    # Forward pass
    rendered1 = renderer1(make_pointcloud())  # (1, H, W, 3)
    rendered2 = renderer2(make_pointcloud())

    # Compute loss
    loss = F.mse_loss(rendered1, target_img1) + F.mse_loss(rendered2, target_img2)
    loss.backward()
    optimizer.step()

    # Visualization & logging
    if i % 50 == 0:
        print(f"Iter {i} | Loss: {loss.item():.4f}")

        # Save rendered images
        img1 = rendered1[0].detach().cpu().numpy()
        img2 = rendered2[0].detach().cpu().numpy()

        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        axs[0].imshow(img1)
        axs[0].set_title("Rendered View 1")
        axs[1].imshow(img2)
        axs[1].set_title("Rendered View 2")
        for ax in axs:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(f"rendered_output/iter_{i:04d}.png")
        plt.close()
        
        
        
        
# --- Save optimized point cloud as binary PLY
optimized_pts = pts.detach().cpu().squeeze(0).numpy()
optimized_colors = colors.detach().cpu().squeeze(0).numpy()

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(optimized_pts)
pcd.colors = o3d.utility.Vector3dVector(optimized_colors)

o3d.io.write_point_cloud("optimized_point_cloud.ply", pcd, write_ascii=False)
print("Saved optimized_point_cloud.ply (binary format)")
