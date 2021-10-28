import os

import trimesh
import glob

files = glob.glob('train_dataset/lm/models/*.ply')
output_path ='train_dataset/lm/simplified_models/'
if os.path.exists(output_path) is False:
    os.makedirs(output_path)

for f in files:
    obj_name = f.split('/')[-1].split('.')[0]
    print(obj_name)
    obj = trimesh.load(f)
    print(obj.faces.shape)
    if obj.faces.shape[0] > 1000:
        obj = obj.simplify_quadratic_decimation(1000)
    obj.export(os.path.join(output_path,f'{obj_name}_simplified.ply'))



# # test
# files = glob.glob('train_dataset/lm/simplified_models/*.ply')
#
# for f in files:
#     obj = trimesh.load(f)
#     print(obj.faces.shape)
#     obj.show()