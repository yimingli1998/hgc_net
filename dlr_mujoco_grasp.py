import mujoco_py
import os
import numpy as np
import time

from mujoco_utils.mjcf_xml import MujocoXML
from mujoco_py import load_model_from_xml, MjSim, functions
from scipy.spatial.transform import Rotation
import copy
import torch
from dataset import GraspDataset
from model import backbone_pointnet2
from matplotlib import pyplot as plt
import math
from mujoco_utils.mj_point_clouds import PointCloudGenerator

from utils import common_util,scene_utils
import random
import json
import trimesh
import glfw
from hitdlr_kinematics.hitdlr_layer.taxonomy_20dof import grasp_dict_20f
import yaml
import loss_utils
with open('config/base_config.yaml', 'r') as f:
    cfg = yaml.load(f,Loader=yaml.FullLoader)

class MujocoEnv:
    def __init__(self):
        self.xml_path ='assets/new_hand_lifting.xml'
        mjb_bytestring = mujoco_py.load_model_from_path(self.xml_path).get_mjb()
        self.model = mujoco_py.load_model_from_mjb(mjb_bytestring)
        self.sim = mujoco_py.MjSim(self.model)
        self.viewer = mujoco_py.MjViewer(self.sim)
        self.scene_xml_path ='assets/scene_xml_temp.xml'
        self.root_path='test_dataset/output/bop_data/lm/train_pbr'
        # self.tax_pose=np.loadtxt('/home/yayu/Documents/dlr_mujoco/dlr_tax_pose.txt')

    def step(self, k=25):
        for _ in range(k):
            self.sim.step()
            self.viewer.render()

    #bulid the scene

    def random_add_obj(self,obj_root_path,num,ranom_pos=True):
        '''
        Adds objs to the secene
        obj_root_path:contanins all the obj_xml files
        num: add n obj to the scene
        '''
        hand_xml = MujocoXML(self.xml_path)
        obj_list=np.random.choice(58,size=num,replace=False)
        for obj in obj_list:
            obj_path=os.path.join(obj_root_path,"obj_"+str(obj).zfill(6)+"_vhacd.xml")
            obj_xml = MujocoXML(obj_path)
            if ranom_pos:
                obj_xml.translate(np.asarray(self.sim.data.get_body_xpos("BHAM.floor")))
                obj_xml.translate([random.uniform(-0.5,0.5),random.uniform(-0.5,0.5),0.01])
                obj_xml.rotate([random.uniform(0,360),random.uniform(0,360),random.uniform(0,360)])
            hand_xml.merge(obj_xml, merge_body=True)
        hand_xml.save_model(self.hand_obj_path)
        xml = hand_xml.get_xml()
        return xml

    def create_scene_obj(self,obj_root_path,index):
        '''
        Creates exactly same scene as ours
        obj_root_path:contanins all the obj_xml files
        index:0~8000
        '''
        hand_xml = MujocoXML(self.xml_path)

        #camera2base
        file_path = os.path.join('test_dataset/output/bop_data/lm/train_pbr',str(index//1000).zfill(6))
        with open(os.path.join(file_path,'scene_gt.json')) as f:
            gt_objs = json.load(f)[str(index%1000)]
        with open(os.path.join(file_path,'scene_camera.json')) as f:
            camera_config = json.load(f)[str(index%1000)]
            print(camera_config.keys())

        R_w2c = np.asarray(camera_config['cam_R_w2c']).reshape(3,3)
        t_w2c = np.asarray(camera_config['cam_t_w2c'])*0.001
        c_w = common_util.inverse_transform_matrix(R_w2c,t_w2c)
        for obj in gt_objs:
            obj_id = obj['obj_id']
            obj_path=os.path.join(obj_root_path,"obj_"+str(obj_id).zfill(6)+"_vhacd.xml")
            obj_xml = MujocoXML(obj_path)

            T_obj = trimesh.transformations.translation_matrix(np.asarray(obj['cam_t_m2c'])*0.001)
            quat_obj = trimesh.transformations.quaternion_from_matrix(np.asarray(obj['cam_R_m2c']).reshape(3,3))
            R_obj = trimesh.transformations.quaternion_matrix(quat_obj)
            matrix_obj = trimesh.transformations.concatenate_matrices(T_obj,R_obj)
            transform = np.dot(c_w,matrix_obj)
            # euler=Rotation.from_quat(quat).as_euler('xyz')
            # euler=euler*180/math.pi
            # # obj_xml.rotate(euler)
            # obj_xml.translate(cam_t_m2c)
            # obj_xml.rotate(euler)
            # obj_xml.translate(np.asarray(self.sim.data.get_body_xpos("floor")))

            c2b_T=transform[0:3,3]
            c2b_R=transform[0:3,0:3]
            c2b_R=Rotation.from_matrix(c2b_R).as_euler('xyz')*180/math.pi
            obj_xml.translate(np.asarray(self.sim.data.get_body_xpos("BHAM.floor")))
            obj_xml.translate(c2b_T)
            obj_xml.rotate(c2b_R)

            hand_xml.merge(obj_xml, merge_body=True)
        hand_xml.save_model(self.scene_xml_path)
        xml = hand_xml.get_xml()
        return xml


    def update_scene_model(self, xml):
        self.model = mujoco_py.load_model_from_xml(xml)
        self.sim = mujoco_py.MjSim(self.model)
        self.viewer = mujoco_py.MjViewer(self.sim)
        

    def set_hand_taxonomy(self,tax):#close hand
        '''
        set hand to certain taxonomy
        '''
        # joint_angle = [0, 0.8, 1,
        #                 0, 1, 1,
        #                0, 1, 1,
        #                0, 1, 1,
        #                0, 1, 1]
        joint_angle=np.asarray(self.tax_pose[tax])
        initialPos =np.asarray(joint_angle)        
        self.sim.data.ctrl[0]=self.sim.data.get_joint_qpos('BHAM.ARTx')
        self.sim.data.ctrl[1]=self.sim.data.get_joint_qpos('BHAM.ARTy')
        self.sim.data.ctrl[2]=self.sim.data.get_joint_qpos('BHAM.ARTz')
        self.sim.data.ctrl[3]=self.sim.data.get_joint_qpos('BHAM.ARRx')
        self.sim.data.ctrl[4]=self.sim.data.get_joint_qpos('BHAM.ARRy')
        self.sim.data.ctrl[5]=self.sim.data.get_joint_qpos('BHAM.ARRz')
        self.sim.data.ctrl[6:]=joint_angle

    def set_hand_pos(self,j,pos,quat):
        '''
         Sets hand to certain position rotation with palm
        '''

        rad = trimesh.transformations.euler_from_quaternion(quat,axes='rxyz')

        joints_angle = np.array([j[0], j[1], j[2],
                                 j[4], j[5], j[6],
                                 j[8], j[9], j[10],
                                 j[12], j[13], j[14],
                                 j[16], j[17], j[18]])

        state = self.sim.get_state()
        state.qpos[0:3] = pos
        state.qpos[3] = rad[0]
        state.qpos[4] = rad[1]
        state.qpos[5] = rad[2]
        state.qpos[6:26] = j

        self.sim.set_state(state)

        self.sim.data.ctrl[0]=pos[0]
        self.sim.data.ctrl[1]=pos[1]
        self.sim.data.ctrl[2]=pos[2]
        self.sim.data.ctrl[3]=rad[0]
        self.sim.data.ctrl[4]=rad[1]
        self.sim.data.ctrl[5]=rad[2]
        self.sim.data.ctrl[6:]=joints_angle

        self.sim.forward()

    def nie(self, j):
        if len(j)==20:
            joints_angle = np.array([j[0], j[1], j[2],
                                     j[4], j[5], j[6],
                                     j[8], j[9], j[10],
                                     j[12], j[13], j[14],
                                     j[16], j[17], j[18]])
        if len(j)==15: in range(400,800):
        evalua
    def qi(self):
        self.sim.data.ctrl[7:9]   += 0.2
        self.sim.data.ctrl[10:12] += 0.2
        self.sim.data.ctrl[13:15] += 0.2
        self.sim.data.ctrl[16:18] += 0.2
        self.sim.data.ctrl[19:21] += 0.2
        self.sim.data.ctrl[2]+=0.5

    def get_pointcloud(self):
        pc_gen = PointCloudGenerator(self.sim, min_bound=(-1., -1., -1.), max_bound=(1., 1., 1.))
        cloud_with_normals,rgb = pc_gen.generateCroppedPointCloud()
        # world_origin_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        # o3d.visualization.draw_geometries([cloud_with_normals, world_origin_axes])  
        return cloud_with_normals,rgb

    def get_depth(self):
        rgb,depth=self.sim.render(1280,720,camera_name="BHAM.fixed",depth=True)
        plt.imshow(rgb[:,:,:3]) # 显示图片
        plt.axis('off') # 不显示坐标轴
        plt.show()
        return depth

    def depthimg2Meters(self, depth):
        extent = self.sim.model.stat.extent
        near = self.sim.model.vis.map.znear * extent
        far = self.sim.model.vis.map.zfar * extent
        image = near / (1 - depth * (1 - near / far))
        return image

    def get_body_mass(self):
        print(self.model.body_mass)

    def get_geom_id(self,name):
        geom_id = self.model.geom_name2id(name)
        return geom_id

    def ray_mesh(self):
        d = functions.mj_rayMesh(self.model,self.sim.data,2,np.asarray([0,0,0]).astype(np.float64),\
                                 np.asarray([0,0,-1]).astype(np.float64))
        return d

    def contact_check(self):
        print('number of contacts', self.sim.data.ncon)
        contact_id = []
        for i in range(self.sim.data.ncon):
            # Note that the contact array has more than `ncon` entries, in range(400,800):
        evalua
            # so be careful to only read the valid entries.
            contact = self.sim.data.contact[i]
            print('contact', i)
            print('dist', contact.dist)
            print('geom1', contact.geom1, self.sim.model.geom_id2name(contact.geom1))
            print('geom2', contact.geom2, self.sim.model.geom_id2name(contact.geom2))
            # There's more stuff in the data structure
            # See the mujoco documentation for more info!
            geom2_body = self.sim.model.geom_bodyid[self.sim.data.contact[i].geom2]
            print(' Contact force on geom2 body', self.sim.data.cfrc_ext[geom2_body])
            print('norm', np.sqrt(np.sum(np.square(self.sim.data.cfrc_ext[geom2_body]))))
            # Use internal functions to read out mj_contactForce
            c_array = np.zeros(6, dtype=np.float64)
            print('c_array', c_array)
            mujoco_py.functions.mj_contactForce(self.sim.model, self.sim.data, i, c_ in range(400,800):
        evaluaarray)
            print('c_array', c_array)
            contact_id.append(contact.geom2)

        if len(contact_id)== 0:
            return True
    def lift_obj(self):
        self.sim.data.ctrl[2]= self.sim.data.ctrl[2]+0.2

    def disable_gravity(self):
        self.model.opt.gravity[-1] = 0

    def open_gravity(self): in range(400,800):
        evalua
        self.model.opt.gravity[-1] = -9.8

    def show_model_info(self):
        """
        Displays relevant model info for the user, namely bodies, joints, actuators, as well as their IDs and ranges.
        Also gives info on which actuators control which joints and which joints are included in the kinematic chain,
        as well as the PID controller info for each actuator.
        """
        print('\nNumber of bodies: {}'.format(self.model.nbody))
        for i in range(self.model.nbody):
            print('Body ID: {}, Body Name: {}'.format(i, self.model.body_id2name(i)))

        print('\nNumber of geoms: {}'.format(self.model.ngeom))
        for i in range(self.model.ngeom):
            print('Gemo ID: {}, Gemo Name: {}'.format(i, self.model.geom_id2name(i)))

        print('\nNumber of joints: {}'.format(self.model.njnt))
        for i in range(self.model.njnt):
            print('Joint ID: {}, Joint Name: {}, Limits: {}'.format(i, self.model.joint_id2name(i), self.model.jnt_range[i]))

        print('\nNumber of Actuators: {}'.format(len(self.sim.data.ctrl)))
        for i in range(len(self.sim.data.ctrl)):
            print('Actuator ID: {},  Controlled Joint: {}, Control Range: {}'.format(i, self.model.actuator_id2name(i), self.model.actuator_ctrlrange[i]))

        print('\n Camera Info: \n')
        for i in range(self.model.ncam):
            print('Camera ID: {}, Camera Name: {}, Camera FOV (y, degrees): {}, Position: {}, Orientation: {}'.format(i, self.model.camera_id2name(i),
                                                                                    self.model.cam_fovy[i], self.model.cam_pos0[i], self.model.cam_mat0[i]))
    def depth_to_pointcloud(self,depth,intrinsic_mat,rgb = None):

        fx, fy = intrinsic_mat[0,0], intrinsic_mat[1,1]
        cx, cy = intrinsic_mat[0,2], intrinsic_mat[1,2]

        xmap, ymap = np.arange(depth.shape[1]), np.arange(depth.shape[0])
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = depth / 1000.0
        points_x = (xmap - cx) / fx * points_z
        points_y = (ymap - cy) / fy * points_z

        mask = points_z > 0
        points_x = points_x[mask]
        points_y = points_y[mask]
        points_z = points_z[mask]
        points = np.stack([points_x, points_y, points_z], axis=-1) in range(400,800):
        evalua

        if rgb is not None:
            points_rgb = rgb[mask]

        else:
            return points,None
        return points,points_rgb

    
    def add_one_obj(self,obj_root_path,obj_t,obj_r,random_pos=True):
        '''
        Adds objs to the secene
        obj_root_path:contanins all the obj_xml files
        num: add n obj to the scene
        '''
        hand_xml = MujocoXML(self.xml_path)  
        obj_xml = MujocoXML(obj_root_path)
        obj_xml.translate(np.asarray(self.sim.data.get_body_xpos("BHAM.floor")))
        if random_pos:
            # obj_t=[random.uniform(-0.25,0.25),random.uniform(-0.25,0.25),0.01]
            # obj_r=[random.uniform(0,360),random.uniform(0,360),random.uniform(0,360)]
            obj_xml.translate(obj_t)
            obj_xml.rotate(obj_r)
        hand_xml.merge(obj_xml, merge_body=True)
        hand_xml.save_model(self.hand_obj_path)
        xml = hand_xml.get_xml()
        return xml


    def get_env_state(self):
        sim_state = self.sim.get_state()
        return sim_state

    def set_env_state(self,state):
        self.sim.set_state(state)
        self.sim.forward()

    def shift_hand(self,pos,quat=[1,0,0,0],euler=None):
        '''
        Sets hand to certain position rotation
        '''
        joints_init = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
       
        initialPos = np.deg2rad(np.asarray(joints_init))
        if not euler :
         euler=Rotation.from_quat(quat).as_euler('xyz')
        sim_state = self.sim.get_state()
        sim_state.qpos[0:3] = pos
        sim_state.qpos[3:6] = euler
        sim_state.qpos[6:26]=initialPos
        self.sim.set_state(sim_state)
        self.sim.forward()

    def show_model_info(self):
        """
        Displays relevant model info for the user, namely bodies, joints, actuators, as well as their IDs and ranges.
        Also gives info on which actuators control which joints and which joints are included in the kinematic chain,
        as well as the PID controller info for each actuator.
        """
        print('\nNumber of bodies: {}'.format(self.model.nbody))
        for i in range(self.model.nbody):
            print('Body ID: {}, Body Name: {}'.format(i, self.model.body_id2name(i)))

        print('\nNumber of geoms: {}'.format(self.model.ngeom))
        for i in range(self.model.ngeom):
            print('Gemo ID: {}, Gemo Name: {}'.format(i, self.model.geom_id2name(i)))

        print('\nNumber of joints: {}'.format(self.model.njnt))
        for i in range(self.model.njnt):
            print('Joint ID: {}, Joint Name: {}, Limits: {}'.format(i, self.model.joint_id2name(i), self.model.jnt_range[i]))

        print('\nNumber of Actuators: {}'.format(len(self.sim.data.ctrl)))
        for i in range(len(self.sim.data.ctrl)):
            print('Actuator ID: {},  Controlled Joint: {}, Control Range: {}'.format(i, self.model.actuator_id2name(i), self.model.actuator_ctrlrange[i]))

        print('\n Camera Info: \n')
        for i in range(self.model.ncam):
            print('Camera ID: {}, Camera Name: {}, Camera FOV (y, degrees): {}, Position: {}, Orientation: {}'.format(i, self.model.camera_id2name(i),
                                                                                                                      self.model.cam_fovy[i], self.model.cam_pos0[i], self.model.cam_mat0[i]))


def eval_model():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    print('Using GPUs ' + os.environ["CUDA_VISIBLE_DEVICES"])
    dataset_path = 'point_grasp_data'
    test_data = GraspDataset(dataset_path,split='test')
    test_dataloader = torch.utils.data.DataLoader(test_data,
                                                  batch_size = 1,
                                                  shuffle=True,
                                                  num_workers = 1)


    model = backbone_pointnet2().cuda()
    model = torch.nn.DataParallel(model)

    model.load_state_dict(torch.load(os.path.join(f"{cfg['eval']['model_path']}/model_{str(cfg['eval']['epoch']).zfill(3)}.pth")))
    model = model.eval()
    taxonomy = ['Parallel_Extension','Pen_Pinch','Palmar_Pinch','Precision_Sphere','Large_Wrap']
    for i, (data,index) in enumerate(test_dataloader):
        bat_point = copy.deepcopy(data['point'])
        img_id = index[0].numpy()

        xml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets/new_hand_lifting.xml')
        env = MujocoEnv()
        scene_xml = env.create_scene_obj('mujoco_objects/objects_xml',img_id)
        env.update_scene_model(scene_xml)
        state = env.get_env_state()

        for k in data.keys():
            data[k] = data[k].cuda().float()
        bat_pred = model(data['point'],data['norm_point'].transpose(1, 2))
        bat_graspable,bat_depth,bat_quat = bat_pred
        bat_pred = torch.cat([bat_graspable,bat_depth.unsqueeze(2),bat_quat],dim = -2)
        for point,pred in zip(bat_point,bat_pred):
            point = point.numpy()
            for j in range(pred.size(-1)): # for each taxonomy
                out = pred[:,:,j].detach().cpu().numpy()
                pos,quat = scene_utils.decode_prediction(point,out,taxonomy[j],img_id,cfg,vis=False)
                for p,q in zip(pos,quat):
                    init_joint = np.asarray(grasp_dict_20f[taxonomy[i]]['joint_init'])*np.pi/180.
                    final_joint = np.asarray(grasp_dict_20f[taxonomy[i]]['joint_final'])*np.pi/180.
                    env.set_hand_pos(j=init_joint, quat=q, pos=p)
                    env.step(100)
                    env.nie(final_joint)
                    env.step(300)
                    env.qi()
                    env.step(1000)
                    env.set_env_state(state)

def eval_bin_model():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    print('Using GPUs ' + os.environ["CUDA_VISIBLE_DEVICES"])
    dataset_path = 'point_grasp_data'
    test_data = GraspDataset(dataset_path,split='test_easy')
    test_dataloader = torch.utils.data.DataLoader(test_data,
                                                  batch_size = 1,
                                                  shuffle=False,
                                                  num_workers = 1)

    model = backbone_pointnet2().cuda()
    model = torch.nn.DataParallel(model)

    model.load_state_dict(torch.load(os.path.join(f"{cfg['eval']['model_path']}/model_{str(cfg['eval']['epoch']).zfill(3)}.pth")))
    model = model.eval()
    taxonomy = ['Parallel_Extension','Pen_Pinch','Palmar_Pinch','Precision_Sphere','Large_Wrap']
    success_dict = {}
    for t in taxonomy:
        success_dict[t] = 0
    for i, (data,index) in enumerate(test_dataloader):
        t0 =time.time()
        bat_point = copy.deepcopy(data['point'])
        img_id = index[0].numpy()
        print(f'scene id:{img_id}')
        scene = trimesh.Scene()
        scene_mesh,gt_objs,_ = scene_utils.load_scene(img_id,split='test')
        num_objs = len(gt_objs)
        # scene.add_geometry(scene_mesh)
        # scene.show()
        xml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets/new_hand_lifting.xml')
        env = MujocoEnv()
        scene_xml = env.create_scene_obj('mujoco_objects/objects_xml',img_id)
        env.update_scene_model(scene_xml)
        state = env.get_env_state()
        init_obj_height = state.qpos[-5:-5-7*(num_objs):-7]
        t1 =time.time()
        for k in data.keys():
            data[k] = data[k].cuda().float()
        bat_pred_graspable,bat_pred_pose,bat_pred_joint = model(data['point'],data['norm_point'].transpose(1, 2))
        for point,gp,pose,joint in zip(bat_point,bat_pred_graspable,bat_pred_pose,bat_pred_joint):
            point_cuda = point.cuda()
            scene = trimesh.Scene()
            scene_mesh,_,_ = scene_utils.load_scene(img_id,split='test')
            scene.add_geometry(scene_mesh)
            scene = scene_utils.add_point_cloud(scene,point,color = cfg['color']['pointcloud'])
            success_rate_list = []
            for t in range(gp.size(-1)): # for each taxonomy
                tax_gp,tax_pose,tax_joint = gp[:,:,t],pose[:,:,t],joint[:,:,t]

                out_gp = torch.argmax(tax_gp,dim = 1).bool()
                if torch.sum(out_gp) > 0:
                    out_gp,out_pos,out_R,out_joint,out_score = loss_utils.decode_pred(point_cuda,tax_gp,tax_pose,tax_joint)
                    out_pos,out_R,out_joint,out_score = out_pos.detach().cpu().numpy(), \
                                              out_R.detach().cpu().numpy(), \
                                              out_joint.detach().cpu().numpy(),\
                                              out_score.detach().cpu().numpy()

                    score_idx = np.argsort(out_score)[::-1]
                    # print(out_score[score_idx])
                    out_pos,out_R,out_joint,out_score = out_pos[score_idx], \
                                                        out_R[score_idx], \
                                                        out_joint[score_idx], \
                                                        out_score[score_idx]
                    good_points = point[out_gp==1]
                    bad_points = point[out_gp==0]
                    # scene = scene_utils.add_point_cloud(scene,good_points,color = cfg['color']['good_point'])
                    # scene = scene_utils.add_point_cloud(scene,bad_points,color = cfg['color']['bad_point'])
                    # scene.show()
                    tax = taxonomy[t]
                    # test gt
                    out_pos,out_quat,out_joint= scene_utils.decode_pred_new(out_pos,out_R,out_joint,tax)
                    topk = 5
                    if len(out_pos)>topk:
                        out_pos,out_quat,out_joint = out_pos[:topk],out_quat[:topk],out_joint[:topk]
                    success =  success_dict[taxonomy[t]]
                    t2 =time.time()
                    for p,q in zip(out_pos,out_quat):
                        init_joint = np.asarray(grasp_dict_20f[taxonomy[t]]['joint_init'])*np.pi/180.
                        final_joint = np.asarray(grasp_dict_20f[taxonomy[t]]['joint_final'])*np.pi/180.
                        env.set_hand_pos(j=init_joint, quat=q, pos=p)
                        env.step(100)
                        env.nie(final_joint)
                        env.step(300)
                        env.qi()
                        # simulate 1000 ste
                        for _ in range (200):
                            env.step(5)
                            cur_state = env.get_env_state().qpos
                            obj_height = cur_state[-5:-5-7*(num_objs):-7]
                            lift_height = obj_height-init_obj_height
                            if np.max(lift_height) >0.2:
                                success +=1
                                break
                            # if cur_state
                        env.set_env_state(state)
                    t3 = time.time()
                    print(t1-t0,'\t',t2-t1,'\t',t3-t2)
                    success_dict[taxonomy[t]] = success

                    success_rate_list.append(success/((i+1)*5.))
            print('success rate:',success_rate_list)
            t3 = time.time()
            print(t1-t0,'\t',t2-t1,'\t',t3-t2)

if __name__ == '__main__':
    # xml_path='./assets/objs/obj_xml/hand_w_obj.xml'
    eval_bin_model()
