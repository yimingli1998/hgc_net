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
from utils import scene_utils, pc_utils
import trimesh
import random

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(CUR_PATH, 'config/base_config.yaml'), 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser('Train DLR Grasp')
parser.add_argument('--batchsize', type=int, default=cfg['train']['batchsize'], help='input batch size')
parser.add_argument('--workers', type=int, default=cfg['train']['workers'], help='number of data loading workers')
parser.add_argument('--epoch', type=int, default=cfg['train']['epoches'], help='number of epochs for training')
parser.add_argument('--gpu', type=str, default=cfg['train']['gpu'], help='specify gpu device')
parser.add_argument('--learning_rate', type=float, default=cfg['train']['learning_rate'],
                    help='learning rate for training')
parser.add_argument('--optimizer', type=str, default=cfg['train']['optimizer'], help='type of optimizer')
parser.add_argument('--theme', type=str, default=cfg['train']['theme'], help='type of train')
parser.add_argument('--model_path', type=str, default=cfg['eval']['model_path'], help='type of train')
FLAGS = parser.parse_args()


# for test
def vis_groundtruth(dataloader):
    taxonomy = ['Parallel_Extension', 'Pen_Pinch', 'Palmar_Pinch', 'Precision_Sphere', 'Large_Wrap']
    for i, (data, index) in enumerate(dataloader):
        img_id = index[0].numpy()
        point = copy.deepcopy(data['point'])[0]
        gt_list = []
        for k in data.keys():
            if data[k].size(-1) == 27:  # graspable 1, depth 1,quat 4 ,metric 1 ,joint 20
                gt_list.append(data[k])
        for i, gt in enumerate(gt_list):
            scene = trimesh.Scene()
            scene_mesh, _, _ = scene_utils.load_scene(img_id)
            scene.add_geometry(scene_mesh)
            pc = trimesh.PointCloud(point, colors=cfg['color']['pointcloud'])
            scene.add_geometry(pc)
            tax = taxonomy[i]
            graspable, pose_label, joint_label = gt[0, :, 0], gt[0, :, 1:5], gt[0, :, 7:]
            gp, pose, joint = point[graspable == 1], pose_label[graspable == 1], joint_label[graspable == 1]
            depth, azimuth, elevation, grasp_angle = pose[:, 0], pose[:, 1], pose[:, 2] + 90, pose[:, 3]
            approach_vector = loss_utils.angle_to_vector(azimuth.double(), elevation.double()).transpose(0,
                                                                                                         1)  # vector B*N, 3
            closing_vector = loss_utils.grasp_angle_to_vector(grasp_angle.double()).transpose(0, 1)
            R = loss_utils.rotation_from_vector(approach_vector, closing_vector)
            pos = gp + (approach_vector * (depth[:, np.newaxis] + 20.) / 100.)
            pos, R, joint = pos.numpy(), R.numpy(), joint.numpy()
            out_pos, out_quat, out_joint = scene_utils.decode_pred_new(pos, R, joint, tax)
            choice = np.random.choice(len(out_pos), 5, replace=True)
            out_pos, out_quat, out_joint = out_pos[choice], out_quat[choice], out_joint[choice]
            for p, q, j in zip(out_pos, out_quat, out_joint):
                # print(p,q,j)
                mat = trimesh.transformations.quaternion_matrix(q)
                hand_mesh = scene_utils.load_hand(p, q, j)
                scene.add_geometry(hand_mesh)
                scene.show()
                break


def vis_model(model, img_idx=0):
    model = model.eval()
    taxonomy = ['Parallel_Extension', 'Pen_Pinch', 'Palmar_Pinch', 'Precision_Sphere', 'Large_Wrap']
    point, sem = scene_utils.load_scene_pointcloud(img_idx, use_base_coordinate=cfg['use_base_coordinate'],
                                                   split='test')
    center = np.mean(point, axis=0)
    norm_point = point - center
    crop_point, crop_index = pc_utils.crop_point(point)
    choice = np.random.choice(len(crop_point), cfg['dataset']['num_points'], replace=False)

    point = point[crop_index][choice]
    sem = sem[crop_index][choice]
    norm_point = norm_point[crop_index][choice]

    # point_cloud = trimesh.PointCloud(point)
    # point_cloud.show()

    bat_point = copy.deepcopy(point)
    bat_sem = copy.deepcopy(sem)
    # for k in data.keys():
    #     data[k] = data[k].cuda().float()
    point = torch.tensor([point]).cuda().float()
    norm_point = torch.tensor([norm_point]).cuda().float()

    bat_pred_graspable, bat_pred_pose, bat_pred_joint = \
        model(point, norm_point.transpose(1, 2))

    point, sem, gp, pose, joint =\
        point[0].cpu(), bat_sem, bat_pred_graspable[0].cpu(), bat_pred_pose[0].cpu(), bat_pred_joint[0].cpu()

    hand_meshes = []
    for t in range(gp.size(-1)):  # for each taxonomy
        tax = taxonomy[t]
        scene = trimesh.Scene()
        scene_mesh, _, _ = scene_utils.load_scene(img_idx, split='test')
        # scene.add_geometry(scene_mesh)
        scene = scene_utils.add_scene_cloud(scene, bat_point)
        # scene.show()
        # exit()
        tax_gp, tax_pose, tax_joint = gp[:, :, t], pose[:, :, t], joint[:, :, t]
        # print(tax_gp.shape,tax_pose.shape,tax_joint.shape)
        if cfg['train']['use_bin_loss']:
            out_gp, out_pos, out_R, out_joint, out_score = loss_utils.decode_pred(point, tax_gp, tax_pose,
                                                                                  tax_joint)
            out_pos, out_R, out_joint, out_score = out_pos.detach().cpu().numpy(), \
                out_R.detach().cpu().numpy(), \
                out_joint.detach().cpu().numpy(), \
                out_score.detach().cpu().numpy()
        else:
            score = F.softmax(tax_gp, dim=1)[:, 1].detach().cpu().numpy()
            tax_gp, tax_pose, tax_joint = tax_gp.detach().cpu().numpy(), tax_pose.detach().cpu().numpy(), tax_joint.detach().cpu().numpy()
            depth, quat = tax_pose[:, 0], tax_pose[:, 1:]
            out_gp = np.argmax(tax_gp, 1)

            mat = trimesh.transformations.quaternion_matrix(quat)
            out_R = mat[:, :3, :3]
            approach = mat[:, :3, 2]
            offset = (depth / 100.0 * (approach.T)).T
            print(offset)
            out_pos = point + offset

            out_pos, out_R, out_joint, out_score = out_pos[out_gp == 1], out_R[out_gp == 1], tax_joint[out_gp == 1], \
            score[out_gp == 1]

        good_points = bat_point[out_gp == 1]
        # bad_points = point[out_gp == 0]
        scene = scene_utils.add_point_cloud(scene, good_points, color=cfg['color'][tax])
        scene.show()
        # exit()
        # scene = scene_utils.add_point_cloud(scene,bad_points,color = cfg['color']['bad_point'])
        for c in np.unique(sem):
            print(c)
            if c > 0.1:
                ins_pos, ins_R, ins_joint, ins_score = out_pos[sem[out_gp == 1] == c], \
                    out_R[sem[out_gp == 1] == c], \
                    out_joint[sem[out_gp == 1] == c], \
                    out_score[sem[out_gp == 1] == c]
                if len(ins_pos) > 0:
                    # ins_pos,ins_R,ins_joint = grasp_utils.grasp_nms(ins_pos,ins_R,ins_joint,ins_score)

                    ins_pos, ins_quat, ins_joint, mask = scene_utils.decode_pred_new(ins_pos, ins_R, ins_joint, tax)
                    # scene = scene_utils.add_point_cloud(scene,point[sem==c],color = [c*20,0,0])
                    for i, (p, q, j) in enumerate(zip(ins_pos, ins_quat, ins_joint)):
                        hand_mesh = scene_utils.load_hand(p, q, j, color=cfg['color'][taxonomy[t]])
                        hand_meshes.append(hand_mesh)
                        # hand_mesh = scene_utils.load_init_hand(p, q, init_hand, color=cfg['color'][taxonomy[t]])
                        scene.add_geometry(hand_mesh)
                        break
        scene.show()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    print('Using GPUs ' + os.environ["CUDA_VISIBLE_DEVICES"])
    # dataset_path = os.path.join(CUR_PATH, '../data/point_grasp_data')
    # train_data = GraspDataset(dataset_path,split='train')
    # train_dataloader = torch.utils.data.DataLoader(train_data,
    #                                                batch_size = FLAGS.batchsize,
    #                                                shuffle=True,
    #                                                num_workers = FLAGS.workers)
    # test_data = GraspDataset(dataset_path,split='test')
    # test_dataloader = torch.utils.data.DataLoader(test_data,
    #                                            batch_size = FLAGS.batchsize,
    #                                            shuffle=True,
    #                                            num_workers = FLAGS.workers)

    model = backbone_pointnet2().cuda()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(
        torch.load(os.path.join(f"{cfg['eval']['model_path']}/model_{str(cfg['eval']['epoch']).zfill(3)}.pth")))
    vis_model(model, 0)
    # vis_groundtruth(train_dataloader)

    # scene = trimesh.Scene()
    # scene_mesh, gt_objs, transform_list = scene_utils.load_scene(0, use_base_coordinate=cfg['use_base_coordinate'],
    #                                                              use_simplified_model=True, split='test')
    # scene.add_geometry(scene_mesh)
    # scene.show()

    # point, sem = scene_utils.load_scene_pointcloud(0, use_base_coordinate=cfg['use_base_coordinate'], split='test')
    # point_cloud = trimesh.PointCloud(sem)
    # point_cloud.show()
