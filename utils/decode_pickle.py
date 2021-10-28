import os
import copy
import pickle
import numpy as np
import torch
import trimesh
from hitdlr_kinematics.hitdlr_layer.hitdlr_layer import HitdlrLayer
from utils import mesh_util
import time
from hitdlr_kinematics.hitdlr_layer.taxonomy_20dof import grasp_dict_20f
import json


# obj_idx_dict = np.load('./obj_idx_dict.npy', allow_pickle=True).item()
# print(obj_idx_dict)
#
# inv_map = {v: k for k, v in obj_idx_dict.items()}
# with open('idx_obj_dict.json', 'w') as fp:
#     json.dump(inv_map, fp, indent=4)
# print(inv_map)
# exit()

vis = True
# mesh_prefix_dir = '/home/v-wewei/code/isaacgym/assets/mjcf/BHAM_split_stl/'
mesh_prefix_dir = '../tmp/picked_obj/'
device = 'cpu'
mesh_save_path = './output_mesh_tmp'
if vis:
    hit_hand = HitdlrLayer(device)
R_hand = np.load('./R_hand.npy')

file_path = '../tmp/pickle_512/'
# file_path = '../../pickle_128'
# file_path = '../pickle'
obj_idx_dict = {}
count = 0
for root, dirs, files in os.walk(file_path):
    for filename in files:
        if filename.endswith('_final.pickle'):

            # if 'Large_Wrap' not in filename:
            #     continue
            grasp_configuration = []
            filepath = os.path.join(root, filename)
            print(filepath)
            # idx = int(filename.split('_')[1])
            # print(idx)

            # if not filename.startswith('D_233_full'):
            #     continue
            with open(filepath, 'rb') as f:
                grasp_dicts = pickle.load(f)
                print(len(grasp_dicts))
                # exit()
                tmp = filepath.split('/')[-1].split('_')[:-3]
                # mesh_filepath = mesh_prefix_dir + '_'.join(tmp) +'_vhacd/' + '_'.join(tmp) +'_smooth.stl'
                obj_filename = '_'.join(tmp) + '.obj'
                mesh_filepath = mesh_prefix_dir + obj_filename
                print(mesh_filepath)
                assert os.path.exists(mesh_filepath)
                # print(mesh_filepath)
                if vis:
                    obj_mesh = mesh_util.Mesh(filepath=mesh_filepath)
                for grasp_dict in grasp_dicts:
                    if not grasp_dict:
                        # print(f'no grasp for {filename}')
                        continue
                    offset = grasp_dict['pos'] - grasp_dict['point'][:3]
                    norm_offset = offset/np.linalg.norm(offset)
                    R = trimesh.transformations.quaternion_matrix(grasp_dict['quat'])[:3,:3]

                    # time_start = time.time()
                    joint_configuration = grasp_dict['joint_configuration']
                    dof = np.asarray(joint_configuration).reshape(5, 4)
                    dof_3 = dof[:, 2]
                    dof_4 = dof[:, 3]
                    dof_min = np.array([min(item1, item2) for item1, item2 in zip(dof_3, dof_4)])
                    dof[:, 2] = dof_min

                    # joint_configuration = dof[:, :3].flatten()
                    pose = np.hstack((grasp_dict['pos'], grasp_dict['quat']))
                    if vis:
                        mesh_copy = copy.deepcopy(obj_mesh)

                    R = trimesh.transformations.quaternion_matrix([pose[3], pose[4], pose[5], pose[6]])

                    t = trimesh.transformations.translation_matrix([pose[0], pose[1], pose[2]])
                    R_obj = trimesh.transformations.concatenate_matrices(t, R)
                    # mesh_copy.apply_transform(R_obj)
                    # mesh_copy.mesh.show()
                    # exit()

                    if vis:
                        theta_ = joint_configuration
                        theta_tensor = torch.from_numpy(theta_).to(device).reshape(-1, 20)

                        pose_tensor = torch.from_numpy(np.identity(4)).to(device).reshape(-1, 4, 4).float()
                        hand_mesh = hit_hand.get_forward_hand_mesh(pose_tensor, theta_tensor, save_mesh=False)
                        hand_mesh = np.sum(hand_mesh)

                    # T_transform_2 = trimesh.transformations.euler_matrix(-0.32, 0, 0, 'rxyz')
                    # T_transform_3 = trimesh.transformations.translation_matrix([-0.015, 0.1, 0.01])
                    # T_transform_4 = trimesh.transformations.quaternion_matrix([0, 1, 0, 0])
                    # T_transform_1 = trimesh.transformations.euler_matrix(np.pi/2, 0, np.pi, 'rxyz')
                    # T_transform_5 = trimesh.transformations.quaternion_matrix([0, 1, 0, 0])
                    # R_hand = trimesh.transformations.concatenate_matrices(T_transform_5,
                    #                                                       T_transform_1, T_transform_4, T_transform_3,
                    #                                                       T_transform_2)

                    inv_R_obj = trimesh.transformations.inverse_matrix(R_obj)
                    hand_in_obj = trimesh.transformations.concatenate_matrices(inv_R_obj, R_hand)

                    translation = copy.deepcopy(hand_in_obj[:3, 3])
                    quat = trimesh.transformations.quaternion_from_matrix(hand_in_obj)
                    # print('time cost is :', time.time() - time_start)
                    if vis:
                        T = trimesh.transformations.quaternion_matrix(quat)
                        # translation_matrix = trimesh.transformations.translation_matrix(translation)
                        # T_ = trimesh.transformations.concatenate_matrices(translation_matrix, T)
                        # print(T, translation_matrix, T_)
                        #
                        # hand_mesh.apply_transform(T_)

                        hand_mesh.apply_transform(T)
                        hand_mesh.apply_translation(translation)
                        hand_mesh.visual.face_colors = [255,255,0]

                        (hand_mesh + mesh_copy.mesh).show()
                        break
                    if obj_filename not in obj_idx_dict.keys():
                        obj_idx_dict[obj_filename] = count
                        count += 1
                    obj_idx = obj_idx_dict[obj_filename]
                    configuration = np.hstack((obj_idx, translation, quat, joint_configuration))

                    grasp_configuration.append(configuration)
            grasp_configuration_array = np.array(grasp_configuration)

            if grasp_configuration_array.shape[0] > 0:
                filename_prefix = filename.split('_')[:-1]
                filename = '_'.join(filename_prefix)
                if not os.path.exists('../new_grasp_dataset_128'):
                    os.mkdir('../new_grasp_dataset_128')
                print(filename)
                np.save('../new_grasp_dataset_128/{}.npy'.format(filename), grasp_configuration_array)
np.save('obj_idx_dict.npy', obj_idx_dict)


