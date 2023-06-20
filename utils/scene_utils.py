import numpy as np
import random
import trimesh
import os
import json
import pickle
from tqdm import tqdm
from hitdlr_kinematics.hitdlr_layer.hitdlr_layer import HitdlrLayer
from hitdlr_kinematics.hitdlr_layer.taxonomy_20dof import grasp_dict_20f
from utils import common_util,pc_utils
import torch
import copy
import yaml
import glob

dir_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(dir_path, '../config/base_config.yaml'), 'r') as f:
    cfg = yaml.load(f,Loader=yaml.FullLoader)

def load_obj(obj_name,color = [255,0,0],transform = None):

    obj = trimesh.load(obj_name)
    if transform:
        obj.apply_transform(transform)
    # obj.visual.face_colors = color
    return obj

def load_init_hand(pos, quat, init_hand,color = [0,0,255]):
    hand_mesh = copy.deepcopy(init_hand)
    T_hand = trimesh.transformations.translation_matrix(pos)
    R_hand = trimesh.transformations.quaternion_matrix(quat)
    matrix_hand = trimesh.transformations.concatenate_matrices(T_hand,R_hand)
    hand_mesh.apply_transform(matrix_hand)
    hand_mesh.visual.face_colors = color
    return hand_mesh

def load_hand(pos, quat, joint_configuration,color = [0,0,255]):

    hit_hand = HitdlrLayer()
    theta_tensor = torch.from_numpy(joint_configuration).reshape(-1, 20)
    # theta_tensor = torch.from_numpy(joint_configuration)
    pose_tensor = torch.from_numpy(np.identity(4)).reshape(-1, 4, 4).float()
    hand_mesh = hit_hand.get_forward_hand_mesh(pose_tensor, theta_tensor, save_mesh=False)
    hand_mesh = np.sum(hand_mesh)
    T_hand = trimesh.transformations.translation_matrix(pos)
    R_hand = trimesh.transformations.quaternion_matrix(quat)
    matrix_hand = trimesh.transformations.concatenate_matrices(T_hand,R_hand)
    hand_mesh.apply_transform(matrix_hand)
    hand_mesh.visual.face_colors = color
    return hand_mesh

def load_scene_pointcloud(img_id, use_base_coordinate=True,split = 'train'):
    if (split =='train' or split =='val'):
        file_path = os.path.join(dir_path,'../../data/train_data/images',str(img_id//1000).zfill(6))
    else:
        file_path = os.path.join(dir_path,'../../data/test_data/images',str(img_id//1000).zfill(6))
        # print('***')
    with open(os.path.join(file_path,'../../camera.json')) as f:
        intrinsics = json.load(f)
    depth_file = os.path.join(file_path,f'depth/{str(img_id%1000).zfill(6)}.png')
    mask_files = glob.glob(os.path.join(file_path,f'mask_visib/{str(img_id%1000).zfill(6)}_*.png'))
    if (split =='train' or split =='val'):
        points,sem = pc_utils.depth_to_pointcloud(depth_file,intrinsics)
    else:
        points,sem = pc_utils.depth_to_pointcloud(depth_file,intrinsics,mask_files)
    if use_base_coordinate:
        # load camera to base pose
        with open(os.path.join(file_path,'scene_camera.json')) as f:
            camera_config = json.load(f)[str(img_id%1000)]
        R_w2c = np.asarray(camera_config['cam_R_w2c']).reshape(3,3)
        t_w2c = np.asarray(camera_config['cam_t_w2c'])*0.001
        c_w = common_util.inverse_transform_matrix(R_w2c,t_w2c)
        points = common_util.transform_points(points,c_w)
    return points,sem

def load_scene(img_id,use_base_coordinate = True,use_simplified_model = False,split = 'train'):
    meshes = []
    if (split =='train' or split =='val'):
        file_path = os.path.join(dir_path, '../../data/train_data/images', str(img_id//1000).zfill(6))
    else:
        file_path = os.path.join(dir_path,'../../data/test_data/images',str(img_id//1000).zfill(6))
        # print('***')
    # load obj poses
    with open(os.path.join(file_path,'scene_gt.json')) as f:
        gt_objs = json.load(f)[str(img_id%1000)]
    # load camera to base pose
    with open(os.path.join(file_path,'scene_camera.json')) as f:
        camera_config = json.load(f)[str(img_id%1000)]
    R_w2c = np.asarray(camera_config['cam_R_w2c']).reshape(3,3)
    t_w2c = np.asarray(camera_config['cam_t_w2c'])*0.001
    c_w = common_util.inverse_transform_matrix(R_w2c,t_w2c)

    # create plannar
    planar = trimesh.creation.box([1,1,0.01])
    planar.visual.face_colors = cfg['color']['plannar']
    if not use_base_coordinate:
        planar.apply_transform(common_util.rt_to_matrix(R_w2c,t_w2c))
    meshes.append(planar)
    transform_list = []
    for obj in gt_objs:
        # print(obj)
        if (split =='train' or split =='val'):
            if use_simplified_model:
                mesh = trimesh.load(
                    os.path.join(dir_path, '../../data/train_data/simplified_models', 'obj_' + str(obj['obj_id']).zfill(6) + '_simplified.ply'))
            else:
                mesh = trimesh.load(os.path.join(dir_path,'../../data/train_data/models','obj_' + str(obj['obj_id']).zfill(6)+'.ply'))
        else:
            mesh = trimesh.load(os.path.join(dir_path,'../../data/test_data/models','obj_' + str(obj['obj_id']).zfill(6)+'.ply'))
        T_obj = trimesh.transformations.translation_matrix(np.asarray(obj['cam_t_m2c'])*0.001)
        quat_obj = trimesh.transformations.quaternion_from_matrix(np.asarray(obj['cam_R_m2c']).reshape(3,3))
        R_obj = trimesh.transformations.quaternion_matrix(quat_obj)
        matrix_obj = trimesh.transformations.concatenate_matrices(T_obj,R_obj)
        mesh.apply_transform(matrix_obj)
        transform = matrix_obj
        if use_base_coordinate:
            mesh.apply_transform(c_w)
            transform = np.dot(c_w,transform)
        transform_list.append(transform)
        mesh.visual.face_colors = cfg['color']['object']
        meshes.append(mesh)
    scene_mesh = np.sum(m for m in meshes)
    return scene_mesh,gt_objs,transform_list

def load_scene_grasp(img_id,taxonomy):
    scene_idx = img_id//cfg['num_images_per_scene']
    file_path = os.path.join(dir_path,'../../data/scene_grasps',f'scene_grasp_{str(scene_idx).zfill(4)}.npy')
    scene_grasp = np.load(file_path,allow_pickle=True).item()
    ungraspable_points = scene_grasp[taxonomy]['0']
    graspable_points = scene_grasp[taxonomy]['1']
    hand_meshes = []
    choice = np.random.choice(len(graspable_points),2,replace = False)
    graspable_points = graspable_points[choice]
    for i,gp in enumerate(graspable_points):
        gp = np.asarray(gp, dtype=np.float32)
        hand_mesh = load_hand(gp[3:6],gp[6:10],gp[11:],color = cfg['color'][taxonomy])
        hand_meshes.append(hand_mesh)
    hand_meshes = np.sum(hand_meshes)
    return hand_meshes

def decode_pickle(obj_name):
    R_hand = np.load(f'{dir_path}/R_hand.npy')
    # sampled_points = np.load(f'sampled_points/{obj_name}_sampled_points.npy')
    sampled_points = np.load(f'tmp/new_sampled_points/{obj_name}_sampled_points.npy')
    taxonomies = grasp_dict_20f.keys()
    single_obj_grasp_dict = {}
    for i,s_p in enumerate(sampled_points):
        single_obj_grasp_dict[i] = {}
        single_obj_grasp_dict[i]['point'] = s_p
        for taxonomy in taxonomies:
            single_obj_grasp_dict[i][taxonomy] = []
    for taxonomy in taxonomies:

        # grasp_file = os.path.join(f'tmp/pickle_obj/{obj_name}_{taxonomy}_final.pickle')
        grasp_file = os.path.join(f'tmp/pickle_512/{obj_name}_{taxonomy}_final.pickle')
        if os.path.exists(grasp_file):
            with open(grasp_file, 'rb') as f:
                grasp_dicts = pickle.load(f)
            for i,grasp_dict in enumerate(grasp_dicts):
                if not grasp_dict:
                    continue
                metric = np.asarray([grasp_dict['metric']])
                joint_configuration = np.asarray(grasp_dict['joint_configuration'])
                pos = np.asarray(grasp_dict['pos'])
                quat = np.asarray(grasp_dict['quat'])

                R = trimesh.transformations.quaternion_matrix(quat)
                t = trimesh.transformations.translation_matrix(pos)
                R_obj = trimesh.transformations.concatenate_matrices(t, R)
                inv_R_obj = trimesh.transformations.inverse_matrix(R_obj)
                hand_in_obj = trimesh.transformations.concatenate_matrices(inv_R_obj, R_hand)
                translation = copy.deepcopy(hand_in_obj[:3, 3])
                quaternion = trimesh.transformations.quaternion_from_matrix(hand_in_obj)

                hand = np.concatenate([translation,quaternion,metric,joint_configuration],axis = -1)
                point = grasp_dict['point']
                # print('point',point)
                # print('pos',pos)
                # exit()

                dist = np.linalg.norm(sampled_points[:,:3] - point[:3],axis =1)
                index = np.argmin(dist)

                if dist[index]==0:
                    single_obj_grasp_dict[index][taxonomy].append(hand)
                else:
                    print('***')

    for i in single_obj_grasp_dict.keys():
        single_obj_grasp_dict[i]['tax_name'] = []
        for taxonomy in taxonomies:
            if single_obj_grasp_dict[i][taxonomy]:
                if taxonomy not in single_obj_grasp_dict[i]['tax_name']:
                    single_obj_grasp_dict[i]['tax_name'].append(taxonomy)
                single_obj_grasp_dict[i][taxonomy] = np.asarray(single_obj_grasp_dict[i][taxonomy])
    return single_obj_grasp_dict

def vis_grasp_dataset(index,cfg):
    point = load_scene_pointcloud(index, use_base_coordinate=cfg['use_base_coordinate'])
    grasp = np.load(f'{dir_path}/../point_grasp_data/scene_{str(index).zfill(6)}_label.npy',allow_pickle=True).item()
    taxonomies = grasp_dict_20f.keys()
    scene = trimesh.Scene()
    if cfg['vis']['vis_scene']:
        scene_mesh,_,_ = load_scene(index)
        # scene.add_geometry(scene_mesh)
        # scene.show()
    if cfg['vis']['vis_pointcloud']:
        pc = trimesh.PointCloud(point,colors = cfg['color']['pointcloud'])
        scene.add_geometry(pc)
    for taxonomy in taxonomies:
        if taxonomy == 'DLR_init':
            bad_points_index = list(grasp[taxonomy].keys())
            bad_point = point[bad_points_index]
            if cfg['vis']['vis_pointcloud']:
                bad_pc =trimesh.PointCloud(bad_point,colors = cfg['color']['bad_point'])
                scene.add_geometry(bad_pc)
        else:
            if grasp[taxonomy]:
                good_points_index = list(grasp[taxonomy].keys())
                good_point = point[good_points_index]
                if cfg['vis']['vis_pointcloud']:
                    good_pc =trimesh.PointCloud(good_point,colors = cfg['color']['good_point'])
                    scene.add_geometry(good_pc)
                if cfg['vis']['vis_handmesh']:
                    for index in good_points_index:
                        hand = grasp[taxonomy][index][3:]
                        hand = np.asarray(hand,dtype = np.float32)
                        hand_mesh = load_hand(hand[:3],hand[3:7],hand[8:],color = cfg['color']['hand_mesh'])
                        scene.add_geometry(hand_mesh)
                        break
        scene.show()

def decode_prediction(point, pred_hand, taxonomy, img_id, cfg,vis = True):
    '''
    :param pred_hand: size:(N*(2+1+4))
    :return:
    '''
    # print(pred_hand.shape)
    R_hand = np.load(os.path.join(dir_path,'R_hand.npy'))
    graspable,depth,quat = pred_hand[:,:2],pred_hand[:,2],pred_hand[:,3:]
    out = np.argmax(graspable,1)
    mask = (out == 1)
    depth, quat = depth[mask], quat[mask]
    good_point = point[mask]

    mat = trimesh.transformations.quaternion_matrix(quat)
    approach = mat[:,:3,2]
    offset = (depth*(approach.T)).T
    pos = good_point + offset
    mat[:,:3,3] = pos

    # right dot R_hand_inv
    new_mat = np.dot(mat,R_hand)
    pos = new_mat[:,:3,3]
    R = new_mat[:,:3,:3]
    quat = common_util.matrix_to_quaternion(R)

    if vis:
        scene = trimesh.Scene()
        pc = trimesh.PointCloud(point,colors = cfg['color']['pointcloud'])
        scene.add_geometry(pc)
        good_pc = trimesh.PointCloud(good_point,colors = cfg['color']['good_point'])
        scene.add_geometry(good_pc)

        good_mask = good_point[:,2] > 0.01 #filter plannar
        pos,quat = pos[good_mask], quat[good_mask]
        init_hand = trimesh.load(f'dir_path/../hand_taxonomy_mesh/{taxonomy}.stl')

        scene_mesh, _, _ = load_scene(img_id,use_base_coordinate = True)
        scene.add_geometry(scene_mesh)

        choice = np.random.choice(len(pos),5,replace=True)
        pos,quat = pos[choice],quat[choice]
        for p,q in zip(pos,quat):
            hand_mesh = load_init_hand(p, q, init_hand,color = cfg['color']['hand_mesh'])
            scene.add_geometry(hand_mesh)
        scene.show()


    return pos,quat

def decode_pred_new(pos,R,joint,tax):
    R_hand = np.load(os.path.join(dir_path,'R_hand.npy'))
    mat = np.tile(np.eye(4),[R.shape[0],1,1])
    mat[:,:3,:3] = R
    mat[:,:3,3] = pos
    # print(R)
    mask = R[:,2,2] > 0
    mat,pos,R,joint = mat[mask],pos[mask],R[mask],joint[mask]
    new_mat = np.dot(mat,R_hand)
    # new_mat = mat
    pos = new_mat[:,:3,3]
    R = new_mat[:,:3,:3]
    quat = common_util.matrix_to_quaternion(R)
    joint_init =  np.asarray(grasp_dict_20f[tax]['joint_init'])*np.pi/180.0
    joint_final =  np.asarray(grasp_dict_20f[tax]['joint_final'])*np.pi/180.0
    joint = joint*(joint_final-joint_init)+joint_init
    # joint = joint*180.0/np.pi
    return pos,quat,joint,mask

def decode_groundtruth(index):
    point = np.load(f'{dir_path}/../point_grasp_data/scene_{str(index).zfill(6)}_point.npy')
    grasp = np.load(f'{dir_path}/../point_grasp_data/scene_{str(index).zfill(6)}_label.npy',allow_pickle=True).item()
    taxonomy_list = ['Parallel_Extension','Pen_Pinch','Palmar_Pinch']
    all_hands = {}
    for taxonomy in taxonomy_list:
        if grasp[taxonomy]:
            hand = grasp[taxonomy].values()
            # print(hand)
    return all_hands

def add_scene_cloud(scene,point):
    bg_point = point[point[:,2]<0.01]
    fg_point = point[point[:,2]>=0.01]
    print(bg_point.shape)
    print(fg_point.shape)
    bg_pc =trimesh.PointCloud(bg_point,colors = cfg['color']['plannar'])
    fg_pc =trimesh.PointCloud(fg_point,colors = cfg['color']['object'])
    scene.add_geometry(bg_pc)
    scene.add_geometry(fg_pc)
    return scene

def add_point_cloud(scene,point,color = [0,255,0]):
    pc =trimesh.PointCloud(point,colors = color)
    scene.add_geometry(pc)
    return scene

def vis_point_data(point_data,cfg,index=None):
    R_hand = np.load(os.path.join(dir_path,'R_hand.npy'))
    scene = trimesh.Scene()
    point = point_data['point']
    fg = point[point[:,2] >0.001]
    pc =trimesh.PointCloud(fg,colors = cfg['color']['pointcloud'])
    scene.add_geometry(pc)
    table = point[point[:,2] <0.001]
    pc_table =trimesh.PointCloud(table,colors = [0,100,0,255])
    # planar = trimesh.creation.box([1,1,0.001])
    # planar.visual.face_colors = [100,0,0,100]
    scene.add_geometry(pc_table)
    if index:
        scene_mesh,_,_ = load_scene(index)
        scene.add_geometry(scene_mesh)
    # scene.show()
    for k in point_data.keys():
        print(k)
        if k!= 'point' and k!= 'norm_point':
            if k=='Parallel_Extension':
            # if k=='Palmar_Pinch':
                label = point_data[k]
                bad_point = point[label[:,0]==0]
                bad_pc = trimesh.PointCloud(bad_point)
                bad_pc = trimesh.PointCloud(bad_point,colors = cfg['color']['bad_point'])
                scene.add_geometry(bad_pc)
                good_point_index = (label[:,0]==1)
                good_point = point[good_point_index]
                # print(good_point.shape)
                label = label[good_point_index]
                good_pc = trimesh.PointCloud(good_point,colors = cfg['color']['good_point'])
                scene.add_geometry(good_pc)
                # for i,lb in enumerate(label):
                #     if cfg['train']['use_bin_loss']:
                #         pass
                #     else:
                #         depth,quat,joint = lb[1],lb[2:6],lb[7:]
                #         R = trimesh.transformations.quaternion_matrix(quat)[:3,:3]
                #         approach = R[:3,2]
                #         offset = depth * approach
                #         pos = good_point[i] + offset
                #
                #         # right dot R_hand
                #         mat = common_util.rt_to_matrix(R,pos)
                #         new_mat = np.dot(mat,R_hand)
                #         new_pos = new_mat[:3,3]
                #         new_quat = trimesh.transformations.quaternion_from_matrix(new_mat)
                #         hand_mesh = load_hand(new_pos,new_quat,joint,color = cfg['color']['hand_mesh'])
                #         scene.add_geometry(hand_mesh)
                #         # print(good_point[i])
                #         # print(label[i,1:4])
                #         if i>5:
                #             break
                scene.show()






