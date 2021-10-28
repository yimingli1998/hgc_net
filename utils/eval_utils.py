import sys
sys.path.append('/home/v-wewei/code/graspit_mujoco')

import os
import time
import trimesh
import numpy as np
import pickle
import glfw
from copy import deepcopy
import copy
import argparse

def set_state(sim, qpos, qvel):
    old_state = sim.get_state()
    new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                     old_state.act, old_state.udd_state)
    sim.set_state(new_state)
    sim.forward()

def init_model(config):
    model = grab_pointnet_v2.CoarseNet(config)
    checkpoint_path = os.path.join(config.save_path, config.checkpoint_name)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.cuda()
    return model

# def init_refine_model(config):
#     model_refine = refinement.RefineNet()
#     checkpoint_path = os.path.join(config.save_path, config.checkpoint_name)
#     state_dict = torch.load(checkpoint_path, map_location='cpu')
#     model.load_state_dict(state_dict)
#     model.cuda()
#     return model


def calculate_metric(hand_mesh, object_mesh):
    # depth = trimesh.proximity.signed_distance(object_mesh, hand_mesh.vertices).max()
    depth = 0
    if depth<0:
        depth = 0
    volume_sum = 0
    # binvoxer_kwargs = {'binvox_path': './binvox/binvox', 'dimension': 128}
    #
    # vg = trimesh.exchange.binvox.voxelize_mesh(hand_mesh, binvoxer=None, export_type='off', **binvoxer_kwargs)

    # vg_hand = trimesh.voxel.creation.local_voxelize(hand_mesh, point=(0, 0, 0), radius=128, pitch=0.001)
    #
    # print(object_mesh.bounding_box.extents)
    # vg_obj = trimesh.voxel.creation.local_voxelize(object_mesh, point=(0, 0, 0), radius=128, pitch=0.001)
    # vg_mesh_hand = vg_hand.marching_cubes
    # vg_mesh_obj = vg_obj.marching_cubes
    # vg_mesh_hand.vertices /= 1000.
    # vg_mesh_obj.vertices /= 1000.
    # print(vg_mesh_obj.bounding_box.extents)

    # collision_manager = trimesh.collision.CollisionManager()
    # collision_manager.add_object('object_mesh', object_mesh)
    # for mesh in hand_mesh.split():
    #
    #     is_collision = collision_manager.in_collision_single(mesh)
    #     if is_collision:
    #         # volume = trimesh.boolean.intersection([mesh, object_mesh], engine='scad').volume * 1e6
    #         volume = mesh.intersection(object_mesh).volume * 1e6
    #         volume_sum += volume
    #     else:
    #         volume_sum += 0
    pitch = 0.005
    obj_vox = object_mesh.voxelized(pitch=0.005)
    obj_points = obj_vox.points
    inside = hand_mesh.contains(obj_points)
    volume_sum = inside.sum() * np.power(pitch, 3)

    return depth, volume_sum


def grasp_in_simulation(pose, theta, object):
    inv_pose = np.linalg.inv(pose.detach().squeeze().cpu().numpy())
    rotation = trimesh.transformations.rotation_matrix(angle=0, direction=(1, 0, 0))
    # inv_rotation = np.linalg.inv(rotation)
    matrix = trimesh.transformations.concatenate_matrices(rotation, inv_pose)
    pos = matrix[:3, 3]

    quat = trimesh.transformations.quaternion_from_matrix(matrix)

    obj_body = object.worldbody.getchildren()[0]
    obj_body.set('pos', '0.5 0 0')
    obj_body.set('quat', '1 0 0 0')
    world = MujocoWorldBase()
    gripper = gripper_factory('DLRDexterousGripper_ori')
    world.merge(gripper)
    world.merge(object)
    model = world.get_model(mode="mujoco_py")
    sim = MjSim(model)
    debug_vis = False
    if debug_vis:
        viewer = MjViewer(sim)
        viewer.vopt.geomgroup[0] = 0  # disable visualization of collision mesh
    sim.model.opt.gravity[-1] = 0.0
    sim.step()

    sim.data.ctrl[0] = 0
    sim.data.ctrl[1:] = theta.detach().squeeze().cpu().numpy()
    for _ in range(300):
        sim.step()
        if debug_vis:
            viewer.render()
    qpos = sim.data.qpos
    qvel = np.zeros(27)
    qpos[21:24] = pos
    qpos[24:] = quat
    set_state(sim, qpos, qvel)

    for _ in range(1):
        sim.step()
        if debug_vis:
            viewer.render()
    height_1 = sim.data.qpos[-6]
    sim.data.ctrl[1:] = theta.detach().squeeze().cpu().numpy() + extra_angle
    for _ in range(250):
        sim.step()
        if debug_vis:
            viewer.render()

    sim.model.opt.gravity[-2] = 9.8
    for _ in range(500):
        sim.step()
        if debug_vis:
            viewer.render()

    height_2 = sim.data.qpos[-6]
    success = True
    if height_2 - height_1 > 0.05:
        success = False
    print(success)
    if debug_vis:
        glfw.destroy_window(viewer.window)

    return success


def test_model(net, object, object_vertices, object_mesh):
    print(object_vertices.shape)
    depth_list = []
    volume_list = []
    success_list = []
    object_vertices_tensor = torch.from_numpy(object_vertices[:, :3].astype(np.float32)).to(device).reshape(1, 2048, 3)
    for i in range(10):
        candidate_points = object_vertices_tensor[0, np.random.choice(object_vertices.shape[0], 1)]
        print(object_vertices_tensor.shape)
        print(candidate_points.shape)

        output = net.sample(object_vertices_tensor)

        candidate_points_array = candidate_points.squeeze().detach().cpu().numpy()
        pose = output['pose']  # .squeeze().detach().cpu().numpy()
        theta = torch.deg2rad(output['hand_joint_configuration'])
        meshes = hit.get_forward_hand_mesh(pose, theta, save_mesh=False, path='./output_mesh')
        vis = True
        if vis:

            pc = trimesh.PointCloud(object_vertices[:, :3], colors=[0, 0, 255])
            point = trimesh.PointCloud(candidate_points_array.reshape(-1, 3), colors=[255, 0, 0])
            scene = trimesh.Scene([pc, meshes[0], point])
            scene.show()
        # calculate penetration depth and volume
        depth, volume = calculate_metric(meshes[0], object_mesh)
        depth_list.append(depth)
        volume_list.append(volume)

        # calculate successful rate
        success = grasp_in_simulation(pose, theta, object)
        success_list.append(success)

    return np.array(depth_list).mean(), np.array(volume_list).mean(), np.array(success_list).mean()


if __name__ == "__main__":

    args = parser.parse_args()
    config = cfg_from_yaml_file(args.cfg, cfg)
    if config.cudnn:
        print('use cudnn')
        torch.backends.cudnn.enabled = True
    else:
        print('do not use cudnn')
        torch.backends.cudnn.enabled = False

    model = init_model(config)
    # refine_model = init_refine_model(config)
    depth_mean_list = []
    volume_mean_list = []
    success_mean_list = []
    mesh_dir = '/home/v-wewei/dataset/test_objects'
    for root, dirs, files in os.walk(mesh_dir):
        for filename in files:
            if filename.endswith('.stl') and '_cvx_' not in filename:
                prefix = filename[:-4]
                filepath = os.path.join(root, filename)
                object = MujocoXMLObject(fname='{}/obj_xml/{}.xml'.format(mesh_dir, prefix), name="{}".format(prefix))
                object_mesh = trimesh.load(filepath)

                object_vertices = np.load(os.path.join('data/test_object_vertices_2048/{}.npy'.format(prefix)))
                # object_vertices = np.load(os.path.join('dataset/object_vertices_2048/{}.npy'.format(prefix)))
                depth_mean, volume_mean, success_mean = test_model(model, object, object_vertices, object_mesh)
                depth_mean_list.append(depth_mean)
                volume_mean_list.append(volume_mean)
                success_mean_list.append(success_mean)
    print(mean(depth_mean_list), mean(volume_mean_list), mean(success_mean_list))






