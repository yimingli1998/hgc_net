import glob
import os
import trimesh
import shutil
import numpy as np

# obj_files = glob.glob(os.path.join('dlr_dataset','*/*full_smooth.obj'))
# output = os.path.join('blender_obj')
# if os.path.exists(output) is False:
#     os.makedirs(output)
# print(f'num objs:{len(obj_files)}')
# for i,src in enumerate(obj_files):
#     obj = trimesh.load(src)
#     print(i)
#     print(obj)
#     obj.show()


# R_hand = np.load(f'utils/R_hand.npy')
# hand = trimesh.load(f'hand_taxonomy_mesh/DLR_init.stl')
# mat = np.asarray([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]])
#
# hand.apply_transform(mat)
# hand.show()
cnt = 0
for s_id in range(0,5000):
    grasp = np.load(
        os.path.join('scene_grasps', f'scene_grasp_{str(s_id).zfill(4)}.npy'),
        allow_pickle = True).item()
    for k in grasp.keys():
        if k!='DLR_init':
            cnt +=grasp[k]['1'].shape[0]
    print(cnt)
    # trimesh.exchange.export.export_mesh(obj,os.path.join(output,f'obj_{str(i).zfill(6)}.ply'))
    # tgt = os.path.join(output,f'obj_{str(i).zfill(6)}.obj')
    # print(tgt)
    # shutil.copyfile(src,tgt,)

    # mesh = trimesh.load(obj)
    # mesh.show()

# # # sort ply
# obj_files = glob.glob(os.path.join('/media/lym/lym/dlr/BlenderProc/scene_dataset/dataset/lm/models','*.ply'))
# print(obj_files)
# for i,obj in enumerate(obj_files):
#     print(i)
#     print(obj)
#     os.rename(obj, os.path.join('/media/lym/lym/dlr/BlenderProc/scene_dataset/dataset/lm/models',f'obj_{str(i).zfill(6)}.ply'))