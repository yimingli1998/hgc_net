import numpy as np
from PIL import Image

def depth_to_pointcloud(depth_file,intrinsics):

    depth = np.array(Image.open(depth_file))*0.001*intrinsics['depth_scale']
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']
    s = 1.0

    xmap, ymap = np.arange(depth.shape[1]), np.arange(depth.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth / s
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    mask = points_z > 0
    points_x = points_x[mask]
    points_y = points_y[mask]
    points_z = points_z[mask]

    points = np.stack([points_x, points_y, points_z], axis=-1)

    # cloud = o3d.geometry.PointCloud()
    # cloud.points = o3d.utility.Vector3dVector(points)
    # cloud.colors = o3d.utility.Vector3dVector(colors)
    # print(points.shape)
    return points
