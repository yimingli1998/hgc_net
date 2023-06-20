import numpy as np
import trimesh
import os
import json
import pickle
from tqdm import tqdm
from hitdlr_kinematics.hitdlr_layer.hitdlr_layer import HitdlrLayer
from hitdlr_kinematics.hitdlr_layer.taxonomy_20dof import grasp_dict_20f
from utils import common_util, scene_utils,pc_utils
import torch
import copy
import time
from scipy import spatial
import yaml

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(CUR_PATH,'config/base_config.yaml'), 'r') as f:
    cfg = yaml.load(f,Loader=yaml.FullLoader)


def match_point_grasp(img_id,save_path):
    scene = trimesh.Scene()
    grasp_path = os.path.join(CUR_PATH,'../data/scene_grasps')
    taxonomies = grasp_dict_20f.keys()
    scene_id = img_id//cfg['num_images_per_scene']
    point,sem = scene_utils.load_scene_pointcloud(img_id, use_base_coordinate=cfg['use_base_coordinate'])
    grasp = np.load(
        os.path.join(grasp_path, f'scene_grasp_{str(scene_id).zfill(4)}.npy'),
        allow_pickle = True).item()
    pc =trimesh.PointCloud(point,colors = [0,255,0])
    scene.add_geometry(pc)
    # scene.show()
    point_grasp_dict = {}
    kdtree = spatial.KDTree(point)
    for taxonomy in taxonomies:
        point_grasp_dict[taxonomy] = {}
        if taxonomy == 'DLR_init':
            all_grasp_point = grasp[taxonomy]['0']
            points_query = kdtree.query_ball_point(all_grasp_point, 0.005)
            points_query = [item for sublist in points_query for item in sublist]
            points_query = list(set(points_query))
            for index in points_query:
                point_grasp_dict[taxonomy][index] = -1
        else:
            if grasp[taxonomy]['1'] != []:
                good_point = grasp[taxonomy]['1'][:,:3]
                points_query = kdtree.query_ball_point(good_point, 0.005)
                for i,pq in enumerate(points_query):
                    if pq != []:
                        for index in pq:
                            point_grasp_dict[taxonomy][index] = grasp[taxonomy]['1'][i]
    np.save(f'{save_path}/scene_{str(img_id).zfill(6)}_point.npy',point)
    np.save(f'{save_path}/scene_{str(img_id).zfill(6)}_label.npy',point_grasp_dict)

    print(f'scene_{str(img_id).zfill(6)} finished!')

def parallel_match_point_grasp(proc,save_path):
    from multiprocessing import Pool
    p = Pool(processes = proc)
    res_list = []
    for i in range(0,cfg['num_images']):
        res_list.append(p.apply_async(match_point_grasp, (i,save_path,)))
    p.close()
    p.join()
    for res in tqdm(res_list):
        res.get()

if  __name__ =='__main__':
    save_path = os.path.join(CUR_PATH,'../data/point_grasp_data')
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    # match_point_grasp(1,save_path)
    parallel_match_point_grasp(6,save_path)

    # # test save file
    # point = np.load(f'{save_path}/scene_000004_point.npy')
    # point_grasp_dict = np.load(f'{save_path}/scene_000004_label.npy',allow_pickle=True).item()
    # taxonomies = grasp_dict_20f.keys()
    # scene = trimesh.Scene()
    # pc =trimesh.PointCloud(point,colors = [0,255,0])
    # scene.add_geometry(pc)
    # for taxonomy in taxonomies:
    #     if taxonomy == 'DLR_init':
    #         bad_points_index = list(point_grasp_dict[taxonomy].keys())
    #         print(bad_points_index)
    #         bad_point = point[bad_points_index]
    #         bad_pc =trimesh.PointCloud(bad_point)
    #         scene.add_geometry(bad_pc)
    #     else:
    #         if point_grasp_dict[taxonomy]:
    #             good_points_index = list(point_grasp_dict[taxonomy].keys())
    #             good_point = point[good_points_index]
    #             good_pc = trimesh.PointCloud(good_point,colors = [255,0,0])
    #             for index in good_points_index:
    #                 hand = point_grasp_dict[taxonomy][index][3:]
    #                 hand = np.asarray(hand,dtype = np.float32)
    #                 hand_mesh = scene_utils.load_hand(hand[:3],hand[3:7],hand[8:])
    #                 scene.add_geometry(hand_mesh)
    #                 break
    #             scene.add_geometry(good_pc)
    #             scene.show()
