import numpy as np
import trimesh
import os
import glob
import json
import pickle
from tqdm import tqdm
from hitdlr_kinematics.hitdlr_layer.hitdlr_layer import HitdlrLayer
from hitdlr_kinematics.hitdlr_layer.taxonomy_20dof import grasp_dict_20f
from utils import common_util,scene_utils
from utils.grasp_utils import Graspdata
import torch
import yaml

with open('config/base_config.yaml', 'r') as f:
    cfg = yaml.load(f,Loader=yaml.FullLoader)

if  __name__ =='__main__':
    # if cfg['vis']:
    #     scene = trimesh.Scene()
    # # obj_file_path = os.path.join('./BlenderProc/scene_dataset/dataset/lm/models')
    # obj_file_path = os.path.join('tmp/picked_obj')
    # obj_files = glob.glob(os.path.join(obj_file_path,'*obj'))
    # # obj_files = glob.glob(os.path.join(obj_file_path,'*ply'))
    # taxonomies = grasp_dict_20f.keys()
    # # save_grasp_path = 'grasp_dataset'
    # save_grasp_path = 'tmp/grasp_dataset'
    # if os.path.exists(save_grasp_path) is False:
    #     os.makedirs(save_grasp_path)
    # bad_obj = []
    # for obj_file in tqdm(obj_files):
    #     obj = scene_utils.load_obj(obj_file)
    #     obj_name = obj_file.split('/')[-1].split('.')[0]
    #     # grasp_dict = scene_utils.decode_pickle(obj_name)
    #     # np.save(os.path.join(save_grasp_path,f'{obj_name}.npy'),grasp_dict)
    #
    #
    #     # for visualization
    #     grasp_dict = np.load(os.path.join(save_grasp_path,f'{obj_name}.npy'),allow_pickle=True).item()
    #     good_points,bad_points = [], []
    #     for k in grasp_dict.keys():
    #         if grasp_dict[k]['tax_name']:
    #             good_points.append(grasp_dict[k]['point'][:3])
    #         else:
    #             bad_points.append(grasp_dict[k]['point'][:3])
    #     good_points = np.asarray(good_points)
    #     bad_points = np.asarray(bad_points)
    #     scene = trimesh.Scene()
    #     if good_points.shape[0]==0:
    #         bad_obj.append(obj_name)
    bad_obj = ['D_140_full_smooth', 'ycb_050_medium_clamp_scaled', 'large_mug', 'D_236_full_smooth', 'mug_1', 'D_146_full_smooth', 'ycb_073-b_lego_duplo_scaled', 'D_103_full_smooth', 'D_230_full_smooth', 'kit_BakingVanilla_scaled']
    for obj in bad_obj:
        os.system(f'rm -r ~/dlr/tmp/picked_obj/{obj}.obj')


        # elif (good_points.shape[0] < 20):
        #     bad_pc =trimesh.PointCloud(bad_points,colors = cfg['color']['bad_point'])
        #     scene.add_geometry(bad_pc)
        #     good_pc =trimesh.PointCloud(good_points,colors = cfg['color']['good_point'])
        #     scene.add_geometry(good_pc)
        #     scene.show()
        # kp = grasp_dict[10]
        # if kp['tax_name']:
        #     # print(kp)
        #     for t in kp['tax_name']:
        #         grasps = kp[t]
        #         # print(grasps.shape)
        #         for g in grasps:
        #             g = np.asarray(g,dtype=np.float32)
        #             hand = scene_utils.load_hand(g[:3],g[3:7],g[8:])
        #             if cfg['vis']:
        #                 scene.add_geometry(obj)
        #                 scene.add_geometry(hand)
        #                 scene.show()
