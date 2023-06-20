import trimesh
import glob
import torch
import numpy as np
import os
from utils import scene_utils
import yaml
import sys

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(CUR_PATH,'config/base_config.yaml'), 'r') as f:
    cfg = yaml.load(f,Loader=yaml.FullLoader)

files = glob.glob(os.path.join(CUR_PATH,'../data/train_data/models/*.ply'))
grasp_path = os.path.join(CUR_PATH,'../data/grasp_dataset')
for f in files:
    obj = trimesh.load(f)
    obj.visual.face_colors = [255,215,0]
    # obj.show()
    obj_name = f.split('/')[-1].split('.')[0]
    grasp_label = np.load(os.path.join(grasp_path,f'{obj_name}.npy'),allow_pickle=True).item()
    tax_grasp = {
        'Parallel_Extension':   [],
        'Pen_Pinch':            [],
        'Palmar_Pinch':         [],
        'Precision_Sphere':     [],
        'Large_Wrap':           []

    }
    for p in grasp_label.keys():
        for t in tax_grasp.keys():
            if grasp_label[p][t] != []:
                tax_grasp[t].append(grasp_label[p][t])
    scene = trimesh.Scene()
    scene.add_geometry(obj)
    for t in tax_grasp.keys():
        if len(tax_grasp[t])>0:
            tax_grasp[t] = np.concatenate(tax_grasp[t],axis = 0)
            choice = np.random.choice(len(tax_grasp[t]),1, replace = True)
            hand_conf = tax_grasp[t][choice]
            hand_conf = np.asarray(hand_conf,dtype=np.float32)[0]
            hand = scene_utils.load_hand(hand_conf[:3],hand_conf[3:7],hand_conf[8:],color = cfg['color'][t])
            scene.add_geometry(hand)
    scene.show()
