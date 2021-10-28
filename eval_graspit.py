import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import os
from tqdm import tqdm
import numpy as np
import argparse
import time
import copy
from dataset import GraspDataset
from model import backbone_pointnet2
import yaml
import loss_utils
from utils import scene_utils,grasp_utils,eval_utils,common_util
from hitdlr_kinematics.hitdlr_layer.taxonomy_20dof import grasp_dict_20f
import trimesh
import glob
import json
from dlr_mujoco_grasp import MujocoEnv
with open('config/base_config.yaml', 'r') as f:
    cfg = yaml.load(f,Loader=yaml.FullLoader)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def evaluate_without_tax_graspit(path,img_id):
    init_joint = np.asarray(grasp_dict_20f['DLR_init']['joint_init'])*np.pi/180.
    print('img id:',img_id)
    taxonomy = ['Parallel_Extension','Pen_Pinch','Palmar_Pinch','Precision_Sphere','Large_Wrap']
    scene = trimesh.Scene()
    scene_mesh,gt_objs,_ = scene_utils.load_scene(img_id,split='test')

    num_objs = len(gt_objs)
    env = MujocoEnv()
    scene_xml = env.create_scene_obj('mujoco_objects/objects_xml',img_id)
    env.update_scene_model(scene_xml)
    state = env.get_env_state()
    init_obj_height = state.qpos[-5:-5-7*(num_objs):-7]
    grasp_path = glob.glob(os.path.join(path,f'{img_id}_*.json'))
    # print(grasp_path)
    success_list,obj_list = [],[]
    for i,grasp in enumerate(grasp_path):
        with open(grasp) as f:
            hand = json.load(f)
        # print(hand)
        if hand !=[]:
            score = [h['epsilon'] for h in hand]
            score_idx = np.argsort(score)[::-1]
            if len(score_idx) >2:
                score_idx = score_idx[:2]
            for s in score_idx:
                h = hand[s]
                pose,joint = h['pose'],h['dofs']
                j = h['dofs']
                p,q = pose[:3],pose[3:]
                q = [q[3],q[0],q[1],q[2]]
                env.set_hand_pos(j=init_joint, quat=q, pos=p)
                env.step(100)
                env.nie(j)
                env.step(300)
                env.qi()
                for _ in range (200):
                    env.step(5)
                    cur_state = env.get_env_state().qpos
                    obj_height = cur_state[-5:-5-7*(num_objs):-7]
                    lift_height = obj_height-init_obj_height
                    if np.max(lift_height) >0.2:
                        success =1
                        obj_list.append(i)
                        break
                    success = 0
                    # if cur_state
                env.set_env_state(state)
                success_list.append(success)
        mean_success = np.mean(success_list)
        completion = len(list(set(obj_list)))/float(num_objs)
        penetration = {
            'success':      mean_success,
            'completion':   completion,
            'obj_list':     obj_list

        }
        print(penetration)
    np.save(f'output_res/{img_id}_res_graspit.npy',penetration)
    return penetration
    # grasp = np.load(os.path.join(path,f'img_{img_id}_quat.npy'),allow_pickle= True).item()
    # # print(grasp.keys())
    # penetration = {}
    # all_grasp = []
    # for i,k in enumerate(grasp.keys()):
    #     pos,quat,joint,sem,score = grasp[k]['pos'],grasp[k]['quat'],grasp[k]['joint'],grasp[k]['obj'],grasp[k]['score']
    #     all_grasp.append(np.concatenate([pos,quat,joint,sem[:,np.newaxis],score[:,np.newaxis],np.asarray([i]*len(pos))[:,np.newaxis]],axis = -1))
    # all_grasp = np.concatenate(all_grasp,axis = 0)
    # pos,quat,joint,sem,score,tax = all_grasp[:,:3],all_grasp[:,3:7],all_grasp[:,7:27],all_grasp[:,27],all_grasp[:,28],all_grasp[:,29]
    # R = trimesh.transformations.quaternion_matrix(quat)[:,:3,:3]
    # depth_list,volume_list,success_list,obj_list = [],[],[],[]
    # for c in np.unique(sem):
    #     # print('c',c)
    #     if c> 0.1: # 0 is background
    #         ins_pos,ins_R,ins_joint,ins_score,ins_tax = pos[sem==c], \
    #                                             R[sem==c], \
    #                                             joint[sem==c], \
    #                                             score[sem==c], \
    #                                             tax[sem==c]
    #         if len(ins_pos) >0:
    #             ins_pos,ins_R,ins_joint,ins_tax = grasp_utils.grasp_nms(ins_pos,ins_R,ins_joint,ins_score,ins_tax)
    #             if len(ins_pos) > 2: # each objec choice two grasp
    #                 ins_pos,ins_R,ins_joint,ins_tax = ins_pos[:2],ins_R[:2],ins_joint[:2],ins_tax[:2]
    #             ins_quat = common_util.matrix_to_quaternion(ins_R)
    #             # ins_pos,ins_quat,ins_joint,mask= scene_utils.decode_pred_new(ins_pos,ins_R,ins_joint,k)
    #             for i,(p,q,j,t) in enumerate(zip(ins_pos,ins_quat,ins_joint,ins_tax)):
    #                 # print('i',i)
    #                 k = taxonomy[int(t)]
    #                 init_hand = trimesh.load(f'hand_taxonomy_mesh/{k}.stl')
    #                 # hand_mesh = scene_utils.load_hand(p, q, j, color=cfg['color'][k])
    #                 # depth, volume_sum = eval_utils.calculate_metric(hand_mesh,scene_mesh)
    #                 hand_mesh = scene_utils.load_init_hand(p, q,init_hand, color=cfg['color'][k])
    #                 depth, volume_sum = eval_utils.calculate_metric(hand_mesh,scene_mesh)
    #                 depth_list.append(depth)
    #                 volume_list.append(volume_sum)
    #                 init_joint = np.asarray(grasp_dict_20f[k]['joint_init'])*np.pi/180.
    #                 final_joint = np.asarray(grasp_dict_20f[k]['joint_final'])*np.pi/180.
    #                 env.set_hand_pos(j=init_joint, quat=q, pos=p)
    #                 env.step(100)
    #                 env.nie(final_joint)
    #                 env.step(300)
    #                 env.qi()
    #                 for _ in range (200):
    #                     env.step(5)
    #                     cur_state = env.get_env_state().qpos
    #                     obj_height = cur_state[-5:-5-7*(num_objs):-7]
    #                     lift_height = obj_height-init_obj_height
    #                     if np.max(lift_height) >0.2:
    #                         success =1
    #                         obj_list.append(c)
    #                         break
    #                     success = 0
    #                     # if cur_state
    #                 env.set_env_state(state)
    #                 success_list.append(success)
    #
    #                     # scene.add_geometry(hand_mesh)
    #         mean_depth = np.mean(depth_list)
    #         mean_volume = np.mean(volume_list)
    #         mean_success = np.mean(success_list)
    #         completion = len(list(set(obj_list)))/float(num_objs)
    #         penetration = {
    #             'depth':        mean_depth,
    #             'volume':       mean_volume,
    #             'success':      mean_success,
    #             'completion':   completion,
    #             'obj_list':     obj_list
    #
    #         }
    # print(penetration)
    # np.save(f'output_res/{img_id}_res_quat.npy',penetration)
    return penetration


def parallel_evaluate(path='output',split = 'test_easy',proc = 8):
    from multiprocessing import Pool
    if  split =='test_easy':
        imgIds = list(range(800,810))
    elif split =='test_medium':
        imgIds = list(range(400,800,4))
    elif split =='test_hard':
        imgIds = list(range(0,400))
    else:
        imgIds = list(range(0,1200))

    from multiprocessing import Pool
    p = Pool(processes = proc)
    res_list = []
    for img_id in imgIds:
        res_list.append(p.apply_async(evaluate_without_tax_graspit, (path,img_id,)))
    p.close()
    p.join()
    output = []
    for res in tqdm(res_list):
        output.append(res.get())
    return output

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    print('Using GPUs ' + os.environ["CUDA_VISIBLE_DEVICES"])
    # test(output_path = 'output',split = 'test_medium')
    # for i in range(400,800,4):
    #     evaluate_without_tax_graspit('json_file',i)
    # output = parallel_evaluate('json_file',split = 'test_medium',proc = 8)
f                             output = [np.load(r,allow_pickle =True).item() for r in res]
    # mean_depth = np.mean([out['depth'] for out in output])
    # mean_volume = np.mean([out['volume'] for out in output])
    mean_success = np.mean([out['success'] for out in output])
    mean_complete = np.mean([out['completion'] for out in output])
    # print(f'depth:{mean_depth}')
    # print(f'volume:{mean_volume}')
    print(f'success:{mean_success}')
    print(f'complete:{mean_complete}')

