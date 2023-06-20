import numpy as np
import trimesh
import os
import json
import pickle
from tqdm import tqdm
from hitdlr_kinematics.hitdlr_layer.hitdlr_layer import HitdlrLayer
from hitdlr_kinematics.hitdlr_layer.taxonomy_20dof import grasp_dict_20f
from utils import common_util, scene_utils
import torch
import copy
import yaml

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(CUR_PATH,'config/base_config.yaml'), 'r') as f:
    cfg = yaml.load(f,Loader=yaml.FullLoader)

def save_taxonomy_hand(hand_path):
    taxonomies = grasp_dict_20f.keys()
    for taxonomy in taxonomies:
        hand_joint = np.asarray(grasp_dict_20f[taxonomy]['joint_init'])*np.pi/180.
        hit_hand = HitdlrLayer()
        theta_tensor = torch.from_numpy(hand_joint).reshape(-1, 20)
        pose_tensor = torch.from_numpy(np.identity(4)).reshape(-1, 4, 4).float()
        hand_mesh = hit_hand.get_forward_hand_mesh(pose_tensor, theta_tensor, save_mesh=False)
        hand_mesh = np.sum(hand_mesh)
        hand_mesh = hand_mesh.simplify_quadratic_decimation(2000)
        trimesh.exchange.export.export_mesh(hand_mesh, os.path.join(hand_path,f'{taxonomy}.stl'))

def collision_checker(index):
    print(f"scene {str(index//cfg['num_images_per_scene'])} begin!\n")
    scene = trimesh.Scene()
    scene_mesh,gt_objs,transform_list = scene_utils.load_scene(index,use_base_coordinate = cfg['use_base_coordinate'],use_simplified_model=True)
    scene.add_geometry(scene_mesh)
    collision_manager,_ = trimesh.collision.scene_to_collision(scene)
    taxonomies = grasp_dict_20f.keys()
    taxonomy_hand = {}
    scene_grasp = {}
    scene_grasp_path = os.path.join(CUR_PATH,'../data/scene_grasps')
    if os.path.exists(scene_grasp_path) is False:
        os.makedirs(scene_grasp_path)
    for taxonomy in taxonomies:
        taxonomy_hand['taxonomy'] = trimesh.load(f'../data/hand_taxonomy_mesh/{taxonomy}.stl')
        scene_grasp[taxonomy] = {'0':[],'1':[]}
    for obj,transform in tqdm(zip(gt_objs,transform_list),total=len(gt_objs),desc= f"collision detection for scene {str(index//cfg['num_images_per_scene'])}"):
        obj_name = str(obj['obj_id'])
        hand_grasps = np.load(os.path.join(CUR_PATH,f'../data/grasp_dataset/obj_{obj_name.zfill(6)}.npy'),allow_pickle=True).item()
        for k_p in hand_grasps.keys():
            # kp_grasps: point,DLR_init,Parallel_Extension,Palmar_Pinch,Precision_Sphere,Large_Wrap,tax_name
            kp_grasps = hand_grasps[k_p]
            for taxonomy in taxonomies:
                if taxonomy not in kp_grasps['tax_name']:
                    scene_kp_point = common_util.transform_points(kp_grasps['point'][:3][np.newaxis,:],transform)[0]
                    scene_grasp[taxonomy]['0'].append(scene_kp_point)
                else:
                    kp_grasps_choice = kp_grasps[taxonomy]
                    if len(kp_grasps_choice) > cfg['max_grasp_per_key_point']:
                        choice = np.random.choice(len(kp_grasps_choice),cfg['max_grasp_per_key_point'])
                        kp_grasps_choice = kp_grasps_choice[choice]
                    for kp_g in kp_grasps_choice:
                        T = kp_g[:3]
                        quat = kp_g[3:7]
                        hand_joint = np.asarray(kp_g[8:])*np.pi/180.
                        R = trimesh.transformations.quaternion_matrix(quat)[:3,:3]
                        mat = common_util.rt_to_matrix(R,T)
                        final_pose = np.dot(transform,mat)
                        final_t = copy.deepcopy(final_pose[:3, 3])
                        final_q = trimesh.transformations.quaternion_from_matrix(final_pose[:3,:3])
                        hand_mesh = scene_utils.load_init_hand(final_t,final_q,taxonomy_hand['taxonomy'])
                        # print(hand_mesh.faces.shape)
                        # hand_mesh = scene_utils.load_hand(final_t,final_q,hand_joint)
                        collision  = collision_manager.in_collision_single(hand_mesh)
                        if not collision:
                            scene_kp_point = common_util.transform_points(kp_grasps['point'][:3][np.newaxis,:],transform)[0]
                            label = np.concatenate([scene_kp_point,final_t,final_q,kp_g[7:]])
                            scene_grasp[taxonomy]['1'].append(label)
                            # scene.add_geometry(hand_mesh)
                            # scene.show()
                            break
                    else:
                        scene_kp_point = common_util.transform_points(kp_grasps['point'][:3][np.newaxis,:],transform)[0]
                        scene_grasp[taxonomy]['0'].append(scene_kp_point)
    for taxonomy in taxonomies:
        un_gp_count = len(scene_grasp[taxonomy]['0'])
        print(f"{taxonomy}:{2560-un_gp_count}/{2560}\t")
        scene_grasp[taxonomy]['0'] = np.asarray(scene_grasp[taxonomy]['0'])
        scene_grasp[taxonomy]['1'] = np.asarray(scene_grasp[taxonomy]['1'])
    np.save(os.path.join(scene_grasp_path,f"scene_grasp_{str(index//cfg['num_images_per_scene']).zfill(4)}.npy"),scene_grasp)
    print(f"scene {str(index//cfg['num_images_per_scene'])} finished!\n")

def parallel_collision_checker(proc):
    from multiprocessing import Pool
    p = Pool(processes = proc)
    res_list = []
    for i in range(0,cfg['num_images']):
        if i % cfg['num_images_per_scene'] == 0:
            res_list.append(p.apply_async(collision_checker, (i,)))
    p.close()
    p.join()
    for res in tqdm(res_list):
        res.get()


if __name__ =='__main__':
    hand_path = os.path.join(CUR_PATH,'../data/hand_taxonomy_mesh')
    if os.path.exists(hand_path) is False:
        os.makedirs(hand_path)
    save_taxonomy_hand(hand_path)
    parallel_collision_checker(proc = 16)
    # collision_checker(1000)

    # scene = trimesh.Scene()
    # scene_mesh,gt_objs,transform_list = scene_utils.load_scene(2000)
    # scene.add_geometry(scene_mesh)
    # taxonomies = grasp_dict_20f.keys()
    # for taxonomy in taxonomies:
    #     if taxonomy =='DLR_init':
    #         continue
    #     hand_meshes = scene_utils.load_scene_grasp(2000,taxonomy)
    #     if hand_meshes:
    #         scene.add_geometry(hand_meshes)
    #     scene.show()

