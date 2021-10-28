import os
import trimesh
import shutil


def obj2ply(root_dir, out_dir):
    for root, dirs, files in os.walk(root_dir):
        for i, file_name in enumerate(files):
            if file_name.split('.')[1] == 'obj':
                name_ = file_name.split('.')[0] + '.npy'
                file_ = os.path.join('../grasp_dataset', name_)
                shutil.copyfile(file_, '../grasp_dataset/'+'obj_' + str(i+179).zfill(6) + '.npy')
                file_path = os.path.join(root_dir, file_name)
                print(file_path)
                mesh = trimesh.load_mesh(file_path)
                ply_name = 'obj_' + str(i+179).zfill(6) + '.ply'
                file_path_obj = os.path.join(out_dir, ply_name)
                mesh.export(file_path_obj)
                print(file_path_obj, 'finish')


if __name__ == '__main__':
    root_dir = '../test_obj'
    out_dir = '../test_dataset/lm'
    obj2ply(root_dir, out_dir)