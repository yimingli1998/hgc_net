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
import loss_utils
from scipy import spatial
import yaml
with open('config/base_config.yaml', 'r') as f:
    cfg = yaml.load(f,Loader=yaml.FullLoader)

class GraspDataset:
    def __init__(self,data_path,split='train'):
        self.data_path = data_path
        self.num_imgs = cfg['num_images']
        self.taxonomies = ['Parallel_Extension','Pen_Pinch','Palmar_Pinch','Precision_Sphere','Large_Wrap']
        self.num_points = cfg['dataset']['num_points']
        self.R_hand = np.load('utils/R_hand.npy')
        self.R_hand_inv = common_util.inverse_transform_matrix(self.R_hand[:3,:3],self.R_hand[:3,3])
        self.vis = False
        self.split = split
        if split =='train':
            self.imgIds = list(range(0,16000))
        elif split =='val':
            self.imgIds = list(range(16000,20000))
        elif split =='test_easy':
            self.imgIds = list(range(800,1200))
        elif split =='test_medium':
            self.imgIds = list(range(400,800))
        elif split =='test_hard':
            self.imgIds = list(range(0,400))
        else:
            self.imgIds = list(range(0,1200))
        # if split =='train':
        #     self.imgIds = list(range(0,100))
        # else:
        #     self.imgIds = list(range(80,100))

    def __getitem__(self, index):
        # index = 200
        img_id = self.imgIds[index]
        point,sem = scene_utils.load_scene_pointcloud(img_id, use_base_coordinate=cfg['use_base_coordinate'],split=self.split)

        grasp = np.load(f'{self.data_path}/scene_{str(img_id).zfill(6)}_label.npy',allow_pickle=True).item()
        if self.vis:
            scene_utils.vis_grasp_dataset(img_id,cfg)

        point_data = {}
        point_data['point'] = point
        if (self.split=='train' or self.split=='val'):
            point_data['sem'] = np.zeros(len(point))
        else:
            point_data['sem'] = sem
        center = np.mean(point,axis=0)
        point_data['norm_point'] = point - center

        grasp_point_label = -np.ones(len(point))

        # random select planar points as negative data
        random_choice = np.random.choice(len(point),
                                         int(cfg['dataset']['sample_planar_rate']*len(point)),
                                         replace=False)
        grasp_point_label[random_choice] = 0

        grasp_point_index = list(grasp['DLR_init'].keys())
        grasp_point_label[grasp_point_index] = 0
        for taxonomy in self.taxonomies:
            point_label = copy.deepcopy(grasp_point_label)
            good_points_index = list(grasp[taxonomy].keys())
            point_label[good_points_index] = 1

            hand_conf = np.zeros((len(point),26))   # N*26
            if grasp[taxonomy]:
                hand_conf_label = np.concatenate(list(grasp[taxonomy].values())).reshape(-1,31)[:,3:]   # N*28
                pos = hand_conf_label[:,:3]
                quat = hand_conf_label[:,3:7]
                mat = trimesh.transformations.quaternion_matrix(quat)
                if len(mat.shape) <3:
                    mat = mat[np.newaxis,:,:]
                mat[:,:3,3] = pos

                # right dot R_hand_inv
                new_mat = np.dot(mat,self.R_hand_inv)
                new_pos = new_mat[:,:3,3]
                new_R = new_mat[:,:3,:3]
                approach = new_R[:,:3,2] # the last column
                offset = new_pos - point[good_points_index]
                depth = np.sum(offset*approach,axis =-1)*100 # m to cm
                if cfg['train']['use_bin_loss']:
                    binormal = new_R[:,:3,1]
                    angle_ = np.arctan2(binormal[:,1], binormal[:,0])
                    angle_[angle_<0] = angle_[angle_<0] + np.pi*2.
                    grasp_angle = angle_ / np.pi * 180
                    azimuth_ = np.arctan2(approach[:,1], approach[:,0])
                    azimuth_[azimuth_<0] = azimuth_[azimuth_<0] + np.pi*2.
                    azimuth_angle = azimuth_ / np.pi * 180
                    elevation_ = np.arctan2(-approach[:,2], np.sqrt(approach[:,0]**2 + approach[:,1]**2)) + np.pi/2.
                    elevation_angle = elevation_ / np.pi * 180
                    # print(depth[0],azimuth_angle[0],elevation_angle[0],grasp_angle[0])
                    depth = depth - 20 # [20,28]->[0,8]
                    hand_conf[good_points_index,0] = depth
                    hand_conf[good_points_index,1] = azimuth_angle
                    hand_conf[good_points_index,2] = elevation_angle
                    hand_conf[good_points_index,3] = grasp_angle
                    # hand_conf[good_points_index,4] = None
                    hand_conf[good_points_index,5:] = hand_conf_label[:,7:]

                    # norm joint
                    joint_init =  np.asarray(grasp_dict_20f[taxonomy]['joint_init'])*np.pi/180.0
                    joint_final =  np.asarray(grasp_dict_20f[taxonomy]['joint_final'])*np.pi/180.0
                    hand_conf[good_points_index,6:] = np.clip(hand_conf[good_points_index,6:],joint_init,joint_final)
                    hand_conf[good_points_index,6:] = (hand_conf[good_points_index,6:]-joint_init)/(joint_final-joint_init+0.00001)
                    # print('joint_final',joint_final/np.pi*180.0)
                    hand_conf = hand_conf[:,:26]

                else:
                    new_quat = common_util.matrix_to_quaternion(new_R)

                    hand_conf[good_points_index,0] = depth
                    hand_conf[good_points_index,1:5] = new_quat

                    hand_conf[good_points_index,5:] = hand_conf_label[:,7:]
                    # depth,quat,metric,joint
                    # norm joint
                    joint_init =  np.asarray(grasp_dict_20f[taxonomy]['joint_init'])*np.pi/180.0
                    joint_final =  np.asarray(grasp_dict_20f[taxonomy]['joint_final'])*np.pi/180.0
                    hand_conf[good_points_index,6:] = np.clip(hand_conf[good_points_index,6:],joint_init,joint_final)
                    hand_conf[good_points_index,6:] = (hand_conf[good_points_index,6:]-joint_init)/(joint_final-joint_init+0.00001)

                    hand_conf = hand_conf[:,:26]

            label = np.concatenate((point_label[:,np.newaxis],hand_conf),axis =-1)
            point_data[taxonomy] = label
        crop_point,crop_index = pc_utils.crop_point(point)
        choice = np.random.choice(len(crop_point),self.num_points,replace=False)
        for k in point_data.keys():
        #     print(k)
            point_data[k] = point_data[k][crop_index][choice]

        # scene_utils.vis_point_data(point_data,cfg,img_id)
        # scene_utils.vis_point_data(point_data,cfg)
        return point_data,img_id

    def __len__(self):
        return len(self.imgIds)
        # return 1

if  __name__ =='__main__':
    data_path = 'point_grasp_data'
    gd = GraspDataset(data_path,split = 'train')
    # gd.__getitem__(180)
    dataloader = torch.utils.data.DataLoader(gd, batch_size=1, shuffle=True,
                                             num_workers=1)
    for i,data in enumerate(dataloader):
        pass