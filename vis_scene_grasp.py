import trimesh
import os
from hitdlr_kinematics.hitdlr_layer.taxonomy_20dof import grasp_dict_20f
from utils import common_util, scene_utils

import yaml

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(CUR_PATH,'config/base_config.yaml'), 'r') as f:
    cfg = yaml.load(f,Loader=yaml.FullLoader)

if  __name__ =='__main__':
    taxonomies = grasp_dict_20f.keys()
    for i in range(cfg['num_scenes']):
        if i%cfg['num_images_per_scene'] == 0:
            scene = trimesh.Scene()
            scene_mesh, _, _ = scene_utils.load_scene(i)
            scene.add_geometry(scene_mesh)
            for taxonomy in taxonomies:
                if taxonomy!='DLR_init':
                    hand_meshes = scene_utils.load_scene_grasp(i,taxonomy)
                    scene.add_geometry(hand_meshes)
            scene.show()
            scene_mesh.show()
            # scene.add_geometry(scene_mesh)
            # scene.show()

        # ungraspable_points,grasp
        # if i%cfg['num_images_per_scene'] == 0:
        #     for taxonomy in taxonomies:
        #         # if taxonomy == 'Parallel_Extension':
        #         if taxonomy == 'Pen_Pinch':
        #         # if taxonomy == 'Palmar_Pinch':pable_points,hand_meshes = scene_utils.load_scene_grasp(i,taxonomy)
        #             # ugpc = trimesh.PointCloud(ungraspable_points,colors = [0,255,0])
        #             # gpc = trimesh.PointCloud(graspable_points,colors = [255,0,0])
        #             # scene.add_geometry(hand_meshes)
        #             # scene.add_geometry(ugpc)
        #             # scene.add_geometry(gpc)
        #             # scene.show()