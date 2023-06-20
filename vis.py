import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import os
import tqdm
import numpy as np
import argparse
import time
import copy
from dataset import GraspDataset
from model import backbone_pointnet2
import yaml
import loss_utils
from utils import scene_utils,grasp_utils,common_util
import trimesh
import random
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(CUR_PATH,'config/base_config.yaml'), 'r') as f:
    cfg = yaml.load(f,Loader=yaml.FullLoader)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser('Train DLR Grasp')
parser.add_argument('--batchsize', type=int, default=cfg['train']['batchsize'], help='input batch size')
parser.add_argument('--workers', type=int, default=cfg['train']['workers'], help='number of data loading workers')
parser.add_argument('--epoch', type=int, default=cfg['train']['epoches'], help='number of epochs for training')
parser.add_argument('--gpu', type=str, default=cfg['train']['gpu'], help='specify gpu device')
parser.add_argument('--learning_rate', type=float, default=cfg['train']['learning_rate'], help='learning rate for training')
parser.add_argument('--optimizer', type=str, default=cfg['train']['optimizer'], help='type of optimizer')
parser.add_argument('--theme', type=str, default=cfg['train']['theme'], help='type of train')
parser.add_argument('--model_path', type=str, default=cfg['eval']['model_path'], help='type of train')
FLAGS = parser.parse_args()

# for test
def vis_groundtruth(dataloader):
    taxonomy = ['Parallel_Extension','Pen_Pinch','Palmar_Pinch','Precision_Sphere','Large_Wrap']
    for i, (data,index) in enumerate(dataloader):
        img_id = index[0].numpy()
        point = copy.deepcopy(data['point'])[0]
        gt_list = []
        for k in data.keys():
            if data[k].size(-1) == 27: # graspable 1, depth 1,quat 4 ,metric 1 ,joint 20
                gt_list.append(data[k])
        for i,gt in enumerate(gt_list):
            scene = trimesh.Scene()
            scene_mesh,_,_ = scene_utils.load_scene(img_id)
            scene.add_geometry(scene_mesh)
            pc =trimesh.PointCloud(point,colors = cfg['color']['pointcloud'])
            scene.add_geometry(pc)
            tax = taxonomy[i]
            graspable, pose_label, joint_label = gt[0,:,0], gt[0,:,1:5],gt[0,:,7:]
            gp,pose,joint = point[graspable==1], pose_label[graspable==1], joint_label[graspable==1]
            depth,azimuth,elevation,grasp_angle = pose[:,0],pose[:,1],pose[:,2]+90,pose[:,3]
            approach_vector = loss_utils.angle_to_vector(azimuth.double(), elevation.double()).transpose(0, 1) #vector B*N, 3
            closing_vector = loss_utils.grasp_angle_to_vector(grasp_angle.double()).transpose(0, 1)
            R = loss_utils.rotation_from_vector(approach_vector, closing_vector)
            pos = gp + (approach_vector *(depth[:,np.newaxis]+20.)/100.)
            pos,R,joint = pos.numpy(),R.numpy(),joint.numpy()
            out_pos,out_quat,out_joint = scene_utils.decode_pred_new(pos,R,joint,tax)
            choice = np.random.choice(len(out_pos),5,replace=True)
            out_pos,out_quat,out_joint = out_pos[choice],out_quat[choice],out_joint[choice]
            for p,q,j in zip(out_pos,out_quat,out_joint):
                # print(p,q,j)
                mat = trimesh.transformations.quaternion_matrix(q)
                hand_mesh = scene_utils.load_hand(p, q,j)
                scene.add_geometry(hand_mesh)
                scene.show()
                break


def vis_model(model,dataloader):
    model = model.eval()
    taxonomy = ['Parallel_Extension','Pen_Pinch','Palmar_Pinch','Precision_Sphere','Large_Wrap']
    taxonomy_hand = {}
    init_hand = trimesh.load(f'../data/hand_taxonomy_mesh/DLR_init.stl')
    for t in taxonomy:
        taxonomy_hand[t] = trimesh.load(f'../data/hand_taxonomy_mesh/{t}.stl')
    for i, (data,index) in enumerate(dataloader):
        bat_point = copy.deepcopy(data['point'])
        bat_sem = copy.deepcopy(data['sem'])
        img_id = index[0].numpy()
        for k in data.keys():
            data[k] = data[k].cuda().float()
        bat_pred_graspable,bat_pred_pose,bat_pred_joint = \
            model(data['point'],data['norm_point'].transpose(1, 2))
        for point,sem,gp,pose,joint in zip(bat_point,bat_sem,bat_pred_graspable,bat_pred_pose,bat_pred_joint):
            point_cuda = point.cuda()
            hand_meshes = []
            for t in range(gp.size(-1)): # for each taxonomy
                tax = taxonomy[t]
                scene = trimesh.Scene()
                scene_mesh,_,_ = scene_utils.load_scene(img_id,split='train')
                # scene.add_geometry(scene_mesh)
                scene = scene_utils.add_scene_cloud(scene,point)
                # scene.show()
                tax_gp,tax_pose,tax_joint = gp[:,:,t],pose[:,:,t],joint[:,:,t]
                # print(tax_gp.shape,tax_pose.shape,tax_joint.shape)
                if cfg['train']['use_bin_loss']:
                    out_gp,out_pos,out_R,out_joint,out_score = loss_utils.decode_pred(point_cuda,tax_gp,tax_pose,tax_joint)
                    out_pos,out_R,out_joint,out_score = out_pos.detach().cpu().numpy(),\
                                              out_R.detach().cpu().numpy(),\
                                              out_joint.detach().cpu().numpy(),\
                                              out_score.detach().cpu().numpy()
                else:
                    score = F.softmax(tax_gp,dim = 1)[:,1].detach().cpu().numpy()
                    tax_gp,tax_pose,tax_joint = tax_gp.detach().cpu().numpy(),tax_pose.detach().cpu().numpy(),tax_joint.detach().cpu().numpy()
                    depth,quat = tax_pose[:,0],tax_pose[:,1:]
                    out_gp = np.argmax(tax_gp,1)

                    mat = trimesh.transformations.quaternion_matrix(quat)
                    out_R = mat[:,:3,:3]
                    approach = mat[:,:3,2]
                    offset = (depth/100.0*(approach.T)).T
                    print(offset)
                    out_pos = point + offset

                    out_pos,out_R,out_joint,out_score = out_pos[out_gp==1],out_R[out_gp==1],tax_joint[out_gp==1],score[out_gp==1]


                good_points = point[out_gp==1]
                bad_points = point[out_gp==0]
                scene = scene_utils.add_point_cloud(scene,good_points,color = cfg['color'][tax])
                scene.show()
                # scene = scene_utils.add_point_cloud(scene,bad_points,color = cfg['color']['bad_point'])
                for c in np.unique(sem):
                    print(c)
                    if c > 0.1:
                        ins_pos,ins_R,ins_joint,ins_score = out_pos[sem[out_gp==1]==c],\
                                                            out_R[sem[out_gp==1]==c],\
                                                            out_joint[sem[out_gp==1]==c],\
                                                            out_score[sem[out_gp==1]==c]
                        if len(ins_pos)>0:
                            # ins_pos,ins_R,ins_joint = grasp_utils.grasp_nms(ins_pos,ins_R,ins_joint,ins_score)

                            ins_pos,ins_quat,ins_joint,mask= scene_utils.decode_pred_new(ins_pos,ins_R,ins_joint,tax)
                            # scene = scene_utils.add_point_cloud(scene,point[sem==c],color = [c*20,0,0])
                            for i,(p,q,j) in enumerate(zip(ins_pos,ins_quat,ins_joint)):
                                hand_mesh = scene_utils.load_hand(p, q, j, color=cfg['color'][taxonomy[t]])
                                hand_meshes.append(hand_mesh)
                                # hand_mesh = scene_utils.load_init_hand(p, q, init_hand, color=cfg['color'][taxonomy[t]])
                                scene.add_geometry(hand_mesh)
                                break

                scene.show()
            # hand_meshes = random.sample(hand_meshes,30)
            # coll_scene = trimesh.Scene()
            # for h in hand_meshes:
            #     collision_manager,_ = trimesh.collision.scene_to_collision(coll_scene)
            #     collision  = collision_manager.in_collision_single(h)
            #     if collision ==False:
            #         scene.add_geometry(h)
            #         coll_scene.add_geometry(h)
            # scene.show()
                # out_pos,out_R,out_joint = grasp_utils.grasp_nms(out_pos,out_R,out_joint,out_score)
                # score_idx = np.argsort(out_score)[::-1]
                # # print(out_score[score_idx])
                # out_pos,out_R,out_joint,out_score = out_pos[score_idx],\
                #                                     out_R[score_idx],\
                #                                     out_joint[score_idx],\
                #                                     out_score[score_idx]
                # test gt
                # out_pos,out_quat,out_joint= scene_utils.decode_pred_new(out_pos,out_R,out_joint,tax)
                # out_pos,out_quat,out_joint = scene_utils.decode_pred_new(out_pos,out_R,out_joint,tax)
                # choice = np.random.choice(len(out_pos),5,replace=True)
                # choice = np.arange(5)
                # out_pos,out_quat,out_joint = out_pos[choice],out_quat[choice],out_joint[choice]
                # for i,(p,q,j) in enumerate(zip(out_pos,out_quat,out_joint)):
                #     mat = trimesh.transformations.quaternion_matrix(q)
                #     # hand_mesh = scene_utils.load_hand(p, q, j, color=cfg['color'][taxonomy[t]])
                #     hand_mesh = scene_utils.load_init_hand(p, q, init_hand, color=cfg['color'][taxonomy[t]])
                #     scene.add_geometry(hand_mesh)
                # scene.show()
                    # break

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    print('Using GPUs ' + os.environ["CUDA_VISIBLE_DEVICES"])
    dataset_path = os.path.join(CUR_PATH,'../data/point_grasp_data')
    train_data = GraspDataset(dataset_path,split='train')
    train_dataloader = torch.utils.data.DataLoader(train_data,
                                                   batch_size = FLAGS.batchsize,
                                                   shuffle=True,
                                                   num_workers = FLAGS.workers)
    test_data = GraspDataset(dataset_path,split='test')
    test_dataloader = torch.utils.data.DataLoader(test_data,
                                               batch_size = FLAGS.batchsize,
                                               shuffle=True,
                                               num_workers = FLAGS.workers)

    model = backbone_pointnet2().cuda()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(os.path.join(f"{cfg['eval']['model_path']}/model_{str(cfg['eval']['epoch']).zfill(3)}.pth")))
    vis_model(model,test_dataloader)
    # vis_groundtruth(train_dataloader)