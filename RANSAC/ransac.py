import open3d as o3d

pcd = o3d.io.read_point_cloud("points.ply")

plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)

[a, b, c, d] = plane_model

in_cloud = pcd.select_by_index(inliers)
out_cloud = pcd.select_by_index(inliers, invert=True)

in_cloud.paint_uniform_color([1.0, 0.0, 0.0])
out_cloud.paint_uniform_color([0.8, 0.8, 0.8])
o3d.visualization.draw_geometries([in_cloud, out_cloud])


