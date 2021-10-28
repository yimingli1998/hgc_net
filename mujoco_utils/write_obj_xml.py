from xml.dom import minidom
import xml.etree.ElementTree as ET
import os
import trimesh
import pybullet as op1
from pybullet_utils import bullet_client
import re
import multiprocessing as mp
import random

dir_path = os.path.dirname(os.path.realpath(__file__))

def normalize_obj():
    root_dir = '/home/yayu/Documents/dlr_mujoco/assets/objs'

    for root, dirs, files in os.walk(root_dir):
        for file_name in files:
            if  file_name.split('.')[1] == 'obj' :
                file_path = os.path.join(root, file_name)
                mesh = trimesh.load_mesh(file_path)
                file_path_obj = file_path.split('.')[0] + '.obj'
                mesh.vertices -= mesh.center_mass
                mesh.export(file_path_obj)
                if not mesh.is_watertight:
                    print(file_name, 'not watertight')


def ply2obj(root_dir,out_dir):
    for root, dirs, files in os.walk(root_dir):
        for file_name in files:
            if  file_name.split('.')[1] == 'ply':
                file_path = os.path.join(root_dir, file_name)
                print(file_path)
                mesh = trimesh.load_mesh(file_path)
                obj_name = file_path.split('/')[-1].split('.')[0]+ '.obj'
                file_path_obj = os.path.join(out_dir,obj_name)
                mesh.export(file_path_obj)
                print(file_path_obj, 'finish')


def obj2ply(root_dir, out_dir):
    for root, dirs, files in os.walk(root_dir):
        for i, file_name in enumerate(files):
            if file_name.split('.')[1] == 'ply':
                file_path = os.path.join(root_dir, file_name)
                print(file_path)
                mesh = trimesh.load_mesh(file_path)
                ply_name = 'obj_' + str(i).zfill(6) + '.ply'
                file_path_obj = os.path.join(out_dir, ply_name)
                mesh.export(file_path_obj)
                print(file_path_obj, 'finish')


def ply2stl(root_dir,out_dir):
    for root, dirs, files in os.walk(root_dir):
        for file_name in files:
            if  file_name.split('.')[1] == 'ply':
                file_path = os.path.join(root_dir, file_name)
                print(file_path)
                mesh = trimesh.load_mesh(file_path)
                obj_name = file_path.split('/')[-1].split('.')[0]+ '.stl'
                file_path_obj = os.path.join(out_dir,f"{obj_name.split('.')[0]}_vhacd",obj_name)
                mesh.export(file_path_obj)
                print(file_path_obj, 'finish')

# def remove_piece_stl():
#     root_dir = '/home/ldh/hand/BHAM_split_stl'
#     # BHAM_obj_path = '/home/ldh/hand/BHAM_vhacd_stl'
#     # if not os.path.exists(BHAM_obj_path):
#     #     os.mkdir(BHAM_obj_path)

#     for root, dirs, files in os.walk(root_dir):
#         for file_name in files:
#             if  file_name.split('.')[1] == 'obj' and 'vhacd' in file_name:
#                 file_path = os.path.join(root, file_name)
#                 os.remove(file_path)


def vhacd(name_in,name_out,name_log):
    #pb_client = bullet_client.BulletClient(op1.DIRECT)
    #pb_client.vhacd(name_in, name_out, name_log, concavity=0.0001, gamma=0.0001, maxNumVerticesPerCH=64, resolution=500000)
    os.system('testVHACD --input {} --output {} --maxhulls 32 --concavity 0.0001 --gamma 0.0001 --maxNumVerticesPerCH 64 --resolution 5000000 --log log.txt'.format(name_in, name_out))


def vhacd_to_piece(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file_name in files:
            if  file_name.split('.')[1] == 'obj' and 'vhacd' in file_name:
                file_name_= file_name.split('.')[0]
                file_path = os.path.join(root, file_name)
                file_path_=os.path.join(root, file_name_)
                if not os.path.exists(file_path_):
                    os.mkdir(file_path_)
                meshes = trimesh.load_mesh(file_path)
                mesh_list = meshes.split()
                for i, mesh in enumerate(mesh_list):
                    # new_file_path = file_path.replace('vhacd.obj', 'cvx_{}.stl'.format(i))
                    new_file_name=file_name_+'cvx_{}.stl'.format(i)
                    new_file_path=os.path.join(file_path_,new_file_name)
                    mesh.export(new_file_path)

                

def create_xml(file_dir_name,root_dir,output_path):#dir_name:obj_000044_vhacd root_dir:'/home/yayu/Documents/dlr_mujoco/assets/objs/obj_000044_vhacd

    # create the file structure
    data = ET.Element('mujoco')
    data.set('model',file_dir_name[:10] )#'OBJ'
    compiler = ET.SubElement(data, 'compiler')
    size = ET.SubElement(data, 'size')
    compiler.set('angle','radian')
    compiler.set('meshdir', '')
    compiler.set('texturedir', '')####what???
    size.set('njmax','500')
    size.set('nconmax','100')
    item_asset = ET.SubElement(data,'asset')

    item_wordbody = ET.SubElement(data,'worldbody')
    item_body = ET.SubElement(item_wordbody,'body')
    item_body.set('name',file_dir_name[:10])
    item_body.set('pos','0 0 0')
    item_body.set('euler','0 0 0')

    ###texture###
    # item_body_2=ET.SubElement(item_body,'body')
    # item_body_2.set('name','OBJ')#file_dir_name[:10]
    # item_geom_2=ET.SubElement(item_body_2,'geom')
    # item_geom_2.set('type','mesh')
    # item_geom_2.set('solimp','0.998 0.998 0.001')
    # item_geom_2.set('density','50')
    # item_geom_2.set('friction','0.95 0.3 0.1')
    # item_geom_2.set('mesh', "mesh_"+file_dir_name[:10])
    # item_geom_2.set('material','mtl_'+file_dir_name[:10])
    
    # item_site_1=ET.SubElement(item_body,'site')
    # item_site_1.set('rgba','0 0 0 0')
    # item_site_1.set('size','0.05')
    # item_site_1.set('pos','0 0 -0.045')
    # item_site_1.set('name','bottom_site')

    # item_site_2=ET.SubElement(item_body,'site')
    # item_site_2.set('rgba','0 0 0 0')
    # item_site_2.set('size','0.05')
    # item_site_2.set('pos','0 0 0.03')
    # item_site_2.set('name','top_site')

    # item_site_3=ET.SubElement(item_body,'site')
    # item_site_3.set('rgba','0 0 0 0')
    # item_site_3.set('size','0.05')
    # item_site_3.set('pos','0.03 0.03 0')
    # item_site_3.set('name','horizontal_radius_site')
     ###texture###

    item_joint_tx = ET.SubElement(item_body,'joint')
    item_joint_tx.set('name',file_dir_name[:10]+'_joint')#add file_dir_name if two obj are needed!
    item_joint_tx.set('type','free')
  


    # item_joint_tx = ET.SubElement(item_body,'joint')
    # item_joint_tx.set('name',file_dir_name[:10]+'_OBJTx')#add file_dir_name if two obj are needed!
    # item_joint_tx.set('pos','0 0 0')
    # item_joint_tx.set('axis','1 0 0')
    # item_joint_tx.set('type','slide')
    # item_joint_tx.set('limited','false')
    # item_joint_tx.set('damping','0')

    # item_joint_ty = ET.SubElement(item_body,'joint')
    # item_joint_ty.set('name',file_dir_name[:10]+'_OBJTy')
    # item_joint_ty.set('pos','0 0 0')
    # item_joint_ty.set('axis','0 1 0')
    # item_joint_ty.set('type','slide')
    # item_joint_ty.set('limited','false')
    # item_joint_ty.set('damping','0')

    # item_joint_tz = ET.SubElement(item_body,'joint')
    # item_joint_tz.set('name',file_dir_name[:10]+'_OBJTz')
    # item_joint_tz.set('pos','0 0 0')
    # item_joint_tz.set('axis','0 0 1')
    # item_joint_tz.set('type','slide')
    # item_joint_tz.set('limited','false')
    # item_joint_tz.set('damping','0')

    # item_joint_rx = ET.SubElement(item_body,'joint')
    # item_joint_rx.set('name',file_dir_name[:10]+'_OBJRx')
    # item_joint_rx.set('pos','0 0 0')
    # item_joint_rx.set('axis','1 0 0')
    # item_joint_rx.set('limited','false')
    # item_joint_rx.set('damping','0')

    # item_joint_ry = ET.SubElement(item_body,'joint')
    # item_joint_ry.set('name',file_dir_name[:10]+'_OBJRy')
    # item_joint_ry.set('pos','0 0 0')
    # item_joint_ry.set('axis','0 1 0')
    # item_joint_ry.set('limited','false')
    # item_joint_ry.set('damping','0')

    # item_joint_rz = ET.SubElement(item_body,'joint')
    # item_joint_rz.set('name',file_dir_name[:10]+'_OBJRz')
    # item_joint_rz.set('pos','0 0 0')
    # item_joint_rz.set('axis','0 0 1')
    # item_joint_rz.set('limited','false')
    # item_joint_rz.set('damping','0')

    for root,dirs,files in os.walk(root_dir):#root_dir:/home/yayu/Documents/dlr_mujoco/assets/objs/obj_000044_vhacd
                
        for file_name in files:
            if 'vhacd' not in file_name and file_name.split('.')[-1]=='stl':

                file_name_ = file_name.split('.')[0]#obj_000000
                file_path = os.path.join(root, file_name)#/home/yayu/Documents/dlr_mujoco/assets/objs/obj_000044_vhacd/obj_000000.stl
                item_mesh = ET.SubElement(item_asset, 'mesh')
                item_mesh.set('name', "mesh_"+file_name_)
                dir_name = file_path.split('/')[-2]#obj_000044_vhacd
           
                item_mesh.set('file', os.path.join(root, file_name))
                item_mesh.set('scale',"1 1 1")#
                ###texture###
                # item_texture=ET.SubElement(item_asset, 'texture')
                # item_texture.set('name','tex_'+file_name_)
                # item_texture.set('file',root_dir+'/'+file_name_[:10]+'.png')
                # # pngs= os.listdir('/home/yayu/Documents/dlr_mujoco/assets/objs/textures/')
                # # item_texture.set('file',os.path.join('/home/yayu/Documents/dlr_mujoco/assets/objs/textures/',random.sample(pngs, 1)[0]))
                # item_texture.set('type','2d')
         

                # item_material=ET.SubElement(item_asset, 'material')
                # item_material.set('name', 'mtl_'+file_name_)
                # item_material.set('reflectance', "0.7")
                # item_material.set('texture', 'tex_'+file_name_)
                #  ###texture###

                print(root_dir+'/'+file_name_[:10]+'.png')
              
                
                item_geom = ET.SubElement(item_body, 'geom')
                item_geom.set('pos','0 0 0')
                item_geom.set('type', 'mesh')
                item_geom.set('density', '0')
                item_geom.set('mesh', 'mesh_'+file_name_)
                item_geom.set('contype','0')
                item_geom.set('conaffinity','0')
                item_geom.set('group','0')
                item_geom.set('friction',"1 0.005 0.0001")
              

   
            if 'cvx'in file_name:
                # print(file_name)
                file_name_ = file_name.split('.')[0]
                file_path = os.path.join(root,file_name)
                item_mesh = ET.SubElement(item_asset, 'mesh')
                item_mesh.set('name',file_name_)
                dir_name = file_path.split('/')[-2]
                item_mesh.set('file', os.path.join(root, file_name))
                item_geom = ET.SubElement(item_body,'geom')
                item_geom.set('type','mesh')
                item_geom.set('density','2500')
                item_geom.set('mesh',file_name_)
                item_geom.set('name', file_name_)
                item_geom.set('group', '3')
                item_geom.set('condim', '4')
                item_geom.set('friction', '10')


    # create a new XML file with the results

    et = ET.ElementTree(data)
    fname = os.path.join(output_path,'{}.xml'.format(file_dir_name))

    et.write(fname, encoding='utf-8', xml_declaration=True)
    x = minidom.parse(fname)
    with open(fname, 'w') as f:
        f.write(x.toprettyxml(indent = '  '))



if __name__ == '__main__':
    # # step 1 ply to obj
    # root_dir = os.path.join(dir_path,'../BlenderProc/scene_dataset/dataset/lm/models/')
    # out_dir = os.path.join(dir_path,'../mujoco_objects/origin')
    # if os.path.exists(out_dir) is False:
    #     os.makedirs(out_dir)
    # # ply2obj(root_dir,out_dir)
    # ply2stl(root_dir,out_dir)
    # exit()

   # #  step 2 vhacd
   # #  pool = mp.Pool(mp.cpu_count())
   #  pool = mp.Pool(14)
   #  root_dir = os.path.join(dir_path,'../mujoco_objects/origin')
   #  for root, dirs, files in os.walk(root_dir):
   #     for file_name in files:
   #         if file_name.split('.')[-1] == 'obj':
   #             name_in = os.path.join(root, file_name)
   #             name_out = name_in.replace('.obj', '_vhacd.obj')
   #             name_log = "log.txt"
   #             if not os.path.exists(name_out):
   #                 pool.apply_async(vhacd,args=(name_in,name_out,name_log,))
   #
   #
   #  pool.close()
   #  pool.join()

    # step 3 to piece
    # root_dir = os.path.join(dir_path,'../mujoco_objects/origin')
    # vhacd_to_piece(root_dir)

   # generate xml
    root_dir = os.path.join(dir_path,'../mujoco_objects/origin')
    output_path = os.path.join(dir_path,'../mujoco_objects/objects_xml')
    if os.path.exists(output_path) is False:
        os.makedirs(output_path)
    for root, dirs, files in os.walk(root_dir):

        for dir_name in dirs:#dir_name:obj_000044_vhacd
            src = os.path.join(root,dir_name)#/home/yayu/Documents/dlr_mujoco/assets/objs/obj_xml/obj_000044_vhacd

            create_xml(dir_name,src,output_path)
            # print('{} is ok'.format(src))
    
    #normalize_obj()
    #stl2obj()
    #remove_piece_stl()
    # vhacd_to_piece()

    #generate vhacd
    # pool = mp.Pool(mp.cpu_count())
    # root_dir = '../BlenderProc/scene_dataset/dataset/lm/models/'
    # for root, dirs, files in os.walk(root_dir):
    #    for file_name in files:
    #        # if file_name.split('.')[0].split('_')[-1] == 'large':
    #        #     print(file_name)
    #        if file_name.split('.')[-1] == 'obj' and 'vhacd' not in file_name :
    #            name_in = os.path.join(root, file_name)
    #            name_out = name_in.replace('.obj', '_vhacd.obj')
    #            name_log = "log.txt"
    #            print(name_in)
    #
    #            if not os.path.exists(name_out):
    #                pool.apply_async(vhacd,args=(name_in,name_out,name_log,))
    
    
    # pool.close()
    # pool.join()
