import numpy as np
from PIL import Image
import glob

def depth_to_pointcloud(depth_file,intrinsics,mask_files = None):
    depth = np.array(Image.open(depth_file))*0.001*intrinsics['depth_scale']
    sem = np.zeros((depth.shape[0],depth.shape[1]))
    if mask_files !=None:
        for f in mask_files:
            mask = np.array(Image.open(f))
            obj_id = f.split('_')[-1].split('.')[0]
            sem[mask>0] = int(obj_id)+1
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
    sem = sem[mask]
    points = np.stack([points_x, points_y, points_z], axis=-1)

    # cloud = o3d.geometry.PointCloud()
    # cloud.points = o3d.utility.Vector3dVector(points)
    # cloud.colors = o3d.utility.Vector3dVector(colors)
    # print(points.shape)
    if mask_files !=None:
        return points,sem
    else:
        return points,None

def crop_point(point):
    val_x = (point[:, 0] > -0.5) & (point[:, 0] < 0.5)
    val_y = (point[:, 1] > -0.5) & (point[:, 1] < 0.5)
    val_z = (point[:, 2] > -0.5) & (point[:, 2] < 0.5)
    val = val_x * val_y * val_z

    return point[val],val
