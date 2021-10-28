# hithand layer for torch
import torch
import math
import trimesh
import glob
import os
import numpy as np
from .taxonomy_20dof import grasp_dict_20f
# All lengths are in mm and rotations in radians


def save_to_mesh(vertices, faces, output_mesh_path=None):
    assert output_mesh_path is not None
    with open(output_mesh_path, 'w') as fp:
        for vert in vertices:
            fp.write('v %f %f %f\n' % (vert[0], vert[1], vert[2]))
        for face in faces+1:
            fp.write('f %d %d %d\n' % (face[0], face[1], face[2]))
    print('Output mesh save to: ', os.path.abspath(output_mesh_path))


class HitdlrLayer(torch.nn.Module):
    def __init__(self, device='cpu'):
        # The forward kinematics equations implemented here are from
        super().__init__()
        self.A0 = torch.tensor(0.0, dtype=torch.float32, device=device)
        self.A1 = torch.tensor(0.0, dtype=torch.float32, device=device)
        self.A2 = torch.tensor(0.001 * 55, dtype=torch.float32, device=device)
        self.A3 = torch.tensor(0.001 * 25, dtype=torch.float32, device=device)
        # self.Dw = torch.tensor(0.001 * 76, dtype=torch.float32, device=device)
        # self.Dw_knuckle = torch.tensor(0.001 * 42, dtype=torch.float32, device=device)
        # self.D3 = torch.tensor(0.001 * 9.5, dtype=torch.float32, device=device)
        # self.D1 = torch.tensor(0.0, dtype=torch.float32, device=device)
        # self.D2 = torch.tensor(0.0, dtype=torch.float32, device=device)
        # self.D3 = torch.tensor(0.0, dtype=torch.float32, device=device)
        self.phi0 = torch.tensor(0.0, dtype=torch.float32, device=device)
        self.phi1 = torch.tensor(0.0, dtype=torch.float32, device=device)
        self.phi2 = torch.tensor(0.0, dtype=torch.float32, device=device)
        self.phi3 = torch.tensor(0.0, dtype=torch.float32, device=device)
        # self.phi3 = torch.tensor(-math.pi/2, dtype=torch.float32, device=device)
        dir_path = os.path.split(os.path.abspath(__file__))[0]

        self.T = torch.from_numpy(np.load(os.path.join(dir_path, './T.npy')).astype(np.float32)).to(device).reshape(-1, 4, 4)
        # self.T_AR = torch.tensor([[0.45621433, -0.54958655, -0.69987364, 0.062569057],
        #                           [0.1048023,  0.81419986,  -0.57104734, 0.044544548],
        #                           [0.88367696, 0.18717161,  0.42904758,  0.080044647],
        #                           [0,         0,         0,        1]], dtype=torch.float32, device=device)

        self.T_AR = torch.tensor([[0.429052, -0.571046, -0.699872, 0.061],
                                  [0.187171,  0.814201,  -0.549586, 0.044],
                                  [0.883675, 0.104806,  0.456218,  0.0885],
                                  [0,         0,         0,        1]], dtype=torch.float32, device=device)

        self.T_BR = torch.tensor([[0.0,      -0.087156,  0.996195, -0.001429881],
                                  [0.0,      -0.996195, -0.087156,  0.036800135],
                                  [1.0,       0.0,       0.0,       0.116743545],
                                  [0,         0,         0,         1]], dtype=torch.float32, device=device)

        self.T_CR = torch.tensor([[0.0,  0.0, 1.0, -0.0026],
                                  [0.0, -1.0, 0.0,  0.01],
                                  [1.0,  0.0, 0.0,  0.127043545],
                                  [0,    0,   0,    1]], dtype=torch.float32, device=device)

        self.T_DR = torch.tensor([[0.0,  0.087152, 0.996195, -0.001429881],
                                  [0.0, -0.996195, 0.087152, -0.016800135],
                                  [1.0,  0.0,      0.0,       0.122043545],
                                  [0,         0,         0,        1]], dtype=torch.float32, device=device)

        self.T_ER = torch.tensor([[0.0,  0.1736479, 0.9848078,  0.002071571],
                                  [0.0, -0.9848078, 0.1736479, -0.043396306],
                                  [1.0,  0.0,      0.0,       0.103043545],
                                  [0,         0,         0,        1]], dtype=torch.float32, device=device)

        # self.pi_0_5 = torch.tensor(math.pi / 2, dtype=torch.float32, device=device)
        self.device = device
        self.meshes = self.load_meshes()

        self.righthand = self.meshes["righthand_base"][0]

        # self.palm_2 = self.meshes["palm_2"][0]
        self.base = self.meshes['base'][0]

        self.proximal = self.meshes['proximal'][0]
        self.medial = self.meshes['medial'][0]
        self.distal = self.meshes['distal'][0]

        self.gripper_faces = [
            self.meshes["righthand_base"][1],  # self.meshes["palm_2"][1],
            self.meshes['base'][1], self.meshes['proximal'][1],
            self.meshes['medial'][1], self.meshes['distal'][1]
        ]

        self.vertice_face_areas = [
            self.meshes["righthand_base"][2],  # self.meshes["palm_2"][2],
            self.meshes['base'][2], self.meshes['proximal'][2],
            self.meshes['medial'][2], self.meshes['distal'][2]
        ]

        self.num_vertices_per_part = [
            self.meshes["righthand_base"][0].shape[0],  # self.meshes["palm_2"][0].shape[0],
            self.meshes['base'][0].shape[0], self.meshes['proximal'][0].shape[0],
            self.meshes['medial'][0].shape[0], self.meshes['distal'][0].shape[0]
        ]
        # r and j are used to calculate the forward kinematics for the barrett Hand's different fingers
        # self.r = [-1, 1, 0]
        # self.j = [1, 1, -1]

    def load_meshes(self):
        mesh_path = os.path.dirname(os.path.realpath(__file__)) + "/../meshes/hitdlr_hand_coarse/*"
        mesh_files = glob.glob(mesh_path)
        mesh_files = [f for f in mesh_files if os.path.isfile(f)]
        meshes = {}
        for mesh_file in mesh_files:
            name = os.path.basename(mesh_file)[:-4]
            mesh = trimesh.load(mesh_file)
            triangle_areas = trimesh.triangles.area(mesh.triangles)
            vert_area_weight = []
            for i in range(mesh.vertices.shape[0]):
                vert_neighour_face = np.where(mesh.faces == i)[0]
                vert_area_weight.append(1000000*triangle_areas[vert_neighour_face].mean())
            temp = torch.ones(mesh.vertices.shape[0], 1).float()
            meshes[name] = [
                torch.cat((torch.FloatTensor(np.array(mesh.vertices)), temp), dim=-1).to(self.device),
                # torch.LongTensor(np.asarray(mesh.faces)).to(self.device),
                mesh.faces,
                # torch.FloatTensor(np.asarray(vert_area_weight)).to(self.device),
                vert_area_weight,
                # torch.FloatTensor(mesh.vertex_normals)
                mesh.vertex_normals,
            ]
        return meshes

    def forward(self, pose, theta):
        """[summary]
        Args:
            pose (Tensor (batch_size x 4 x 4)): The pose of the base link of the hand as a translation matrix.
            theta (Tensor (batch_size x 20)): The seven degrees of freedome of the Barrett hand. The first column specifies the angle between
            fingers F1 and F2,  the second to fourth column specifies the joint angle around the proximal link of each finger while the fifth
            to the last column specifies the joint angle around the distal link for each finger

       """
        batch_size = pose.shape[0]

        # rot_z_90 = torch.eye(4, device=self.device)
        #
        # rot_z_90[1, 1] = -1
        # rot_z_90[2, 3] = -0.001 * 79
        # rot_z_90 = rot_z_90.repeat(batch_size, 1, 1)
        # pose = torch.matmul(pose, rot_z_90)
        righthand_vertices = self.righthand.repeat(batch_size, 1, 1)
        righthand_vertices = torch.matmul(torch.matmul(pose, self.T),
                                          righthand_vertices.transpose(2, 1)).transpose(
                                          1, 2)[:, :, :3]
        # palm_1_vertices = self.palm_1.repeat(batch_size, 1, 1)
        # palm_2_vertices = self.palm_2.repeat(batch_size, 1, 1)
        # palm_1_vertices = torch.matmul(pose,
        #                                palm_1_vertices.transpose(2, 1)).transpose(
        #                                  1, 2)[:, :, :3]
        # palm_2_vertices = torch.matmul(pose,
        #                                palm_2_vertices.transpose(2, 1)).transpose(
        #                                  1, 2)[:, :, :3]

        all_base_vertices = torch.zeros(
            (batch_size, 5, self.base.shape[0], 3), device=self.device)      # 5
        all_proximal_vertices = torch.zeros(
            (batch_size, 5, self.proximal.shape[0], 3), device=self.device)  # 5
        all_medial_vertices = torch.zeros(
            (batch_size, 5, self.medial.shape[0], 3), device=self.device)  # 5
        all_distal_vertices = torch.zeros(
            (batch_size, 5, self.distal.shape[0], 3), device=self.device)  # 5

        base_vertices = self.base.repeat(batch_size, 1, 1)
        proximal_vertices = self.proximal.repeat(batch_size, 1, 1)
        medial_vertices = self.medial.repeat(batch_size, 1, 1)
        distal_vertices = self.distal.repeat(batch_size, 1, 1)

        for i in range(5):  # 5
            # print('i is :', i)
            # print(self.A0)
            # print(theta[:, 0+i*4])
            # print(self.phi0)
            # # print(theta[:, 0+i*4]+self.phi0)
            # exit()
            T01 = self.forward_kinematics(self.A0, torch.tensor(0, dtype=torch.float32, device=self.device),
                                          0, -theta[:, 0+i*4]+self.phi0, batch_size)
            T12 = self.forward_kinematics(self.A1, torch.tensor(math.pi/2, dtype=torch.float32, device=self.device),
                                          0, theta[:, 1+i*4]+self.phi1, batch_size)
            # T12 = self.forward_kinematics(self.A1, torch.tensor(0, dtype=torch.float32, device=self.device),
            #                               0, theta[:, 1+i*4]+self.phi1, batch_size)
            T23 = self.forward_kinematics(self.A2, torch.tensor(0, dtype=torch.float32, device=self.device),
                                          0, theta[:, 2+i*4]+self.phi2, batch_size)
            T34 = self.forward_kinematics(self.A3, torch.tensor(0, dtype=torch.float32, device=self.device),
                                          0, theta[:, 3+i*4]+self.phi3, batch_size)

            if i == 0:
                pose_to_Tw0 = torch.matmul(pose, torch.matmul(self.T, self.T_AR))
            elif i == 1:
                pose_to_Tw0 = torch.matmul(pose,  torch.matmul(self.T, self.T_BR))
            elif i == 2:
                pose_to_Tw0 = torch.matmul(pose,  torch.matmul(self.T, self.T_CR))
            elif i == 3:
                pose_to_Tw0 = torch.matmul(pose,  torch.matmul(self.T, self.T_DR))
            elif i == 4:
                pose_to_Tw0 = torch.matmul(pose,  torch.matmul(self.T, self.T_ER))

            pose_to_T01 = torch.matmul(pose_to_Tw0, T01)
            # print('matrix shape is :', pose_to_T01.shape)
            # print('shape is :', base_vertices.shape)
            all_base_vertices[:, i] = torch.matmul(
                pose_to_T01,
                base_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

            pose_to_T12 = torch.matmul(pose_to_T01, T12)

            all_proximal_vertices[:, i] = torch.matmul(
                pose_to_T12,
                proximal_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

            pose_to_T23 = torch.matmul(pose_to_T12, T23)
            all_medial_vertices[:, i] = torch.matmul(
                pose_to_T23,
                medial_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

            pose_to_T34 = torch.matmul(pose_to_T23, T34)
            all_distal_vertices[:, i] = torch.matmul(
                pose_to_T34,
                distal_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        return righthand_vertices, all_base_vertices, all_proximal_vertices, all_medial_vertices, \
               all_distal_vertices

    def forward_kinematics(self, A, alpha, D, theta, batch_size=1):
        c_theta = torch.cos(theta)
        s_theta = torch.sin(theta)
        c_alpha = torch.cos(alpha)
        s_alpha = torch.sin(alpha)
        l_1_to_l = torch.zeros((batch_size, 4, 4), device=self.device)
        l_1_to_l[:, 0, 0] = c_theta
        l_1_to_l[:, 0, 1] = -s_theta
        l_1_to_l[:, 0, 3] = A
        l_1_to_l[:, 1, 0] = s_theta * c_alpha
        l_1_to_l[:, 1, 1] = c_theta * c_alpha
        l_1_to_l[:, 1, 2] = -s_alpha
        l_1_to_l[:, 1, 3] = -s_alpha * D
        l_1_to_l[:, 2, 0] = s_theta * s_alpha
        l_1_to_l[:, 2, 1] = c_theta * s_alpha
        l_1_to_l[:, 2, 2] = c_alpha
        l_1_to_l[:, 2, 3] = c_alpha * D
        l_1_to_l[:, 3, 3] = 1
        return l_1_to_l

    def get_hand_mesh(self, vertices_list, faces, save_mesh=True, path='./output_mesh'):
        if save_mesh:
            assert os.path.exists(path)

        righthand_verts = vertices_list[0]
        righthand_faces = faces[0]
        # palm_2_verts = vertices_list[1]
        # palm_2_faces = faces[1]
        # save_to_mesh(palm_2_verts, palm_2_faces, '{}/hitdlr_palm2.obj'.format(path))

        thumb_base_verts = vertices_list[1][0]
        thumb_base_faces = faces[1]

        thumb_proximal_verts = vertices_list[2][0]
        thumb_proximal_faces = faces[2]

        thumb_medial_verts = vertices_list[3][0]
        thumb_medial_faces = faces[3]

        thumb_distal_verts = vertices_list[4][0]
        thumb_distal_faces = faces[4]

        fore_base_verts = vertices_list[1][1]
        fore_base_faces = faces[1]

        fore_proximal_verts = vertices_list[2][1]
        fore_proximal_faces = faces[2]

        fore_medial_verts = vertices_list[3][1]
        fore_medial_faces = faces[3]

        fore_distal_verts = vertices_list[4][1]
        fore_distal_faces = faces[4]

        middle_base_verts = vertices_list[1][2]
        middle_base_faces = faces[1]

        middle_proximal_verts = vertices_list[2][2]
        middle_proximal_faces = faces[2]

        middle_medial_verts = vertices_list[3][2]
        middle_medial_faces = faces[3]

        middle_distal_verts = vertices_list[4][2]
        middle_distal_faces = faces[4]

        ring_base_verts = vertices_list[1][3]
        ring_base_faces = faces[1]

        ring_proximal_verts = vertices_list[2][3]
        ring_proximal_faces = faces[2]

        ring_medial_verts = vertices_list[3][3]
        ring_medial_faces = faces[3]

        ring_distal_verts = vertices_list[4][3]
        ring_distal_faces = faces[4]

        little_base_verts = vertices_list[1][4]
        little_base_faces = faces[1]

        little_proximal_verts = vertices_list[2][4]
        little_proximal_faces = faces[2]

        little_medial_verts = vertices_list[3][4]
        little_medial_faces = faces[3]

        little_distal_verts = vertices_list[4][4]
        little_distal_faces = faces[4]

        if save_mesh:
            save_to_mesh(righthand_verts, righthand_faces, '{}/hitdlr_righthand.obj'.format(path))
            save_to_mesh(thumb_base_verts, thumb_base_faces, '{}/hitdlr_thumb_base.obj'.format(path))
            save_to_mesh(thumb_proximal_verts, thumb_proximal_faces, '{}/hitdlr_thumb_proximal.obj'.format(path))
            save_to_mesh(thumb_medial_verts, thumb_medial_faces, '{}/hitdlr_thumb_medial.obj'.format(path))
            save_to_mesh(thumb_distal_verts, thumb_distal_faces, '{}/hitdlr_thumb_distal.obj'.format(path))
            save_to_mesh(fore_base_verts, fore_base_faces, '{}/hitdlr_fore_base.obj'.format(path))
            save_to_mesh(fore_proximal_verts, fore_proximal_faces, '{}/hitdlr_fore_proximal.obj'.format(path))
            save_to_mesh(fore_medial_verts, fore_medial_faces, '{}/hitdlr_fore_medial.obj'.format(path))
            save_to_mesh(fore_distal_verts, fore_distal_faces, '{}/hitdlr_fore_distal.obj'.format(path))
            save_to_mesh(middle_base_verts, middle_base_faces, '{}/hitdlr_middle_base.obj'.format(path))
            save_to_mesh(middle_proximal_verts, middle_proximal_faces, '{}/hitdlr_middle_proximal.obj'.format(path))
            save_to_mesh(middle_medial_verts, middle_medial_faces, '{}/hitdlr_middle_medial.obj'.format(path))
            save_to_mesh(middle_distal_verts, middle_distal_faces, '{}/hitdlr_middle_distal.obj'.format(path))
            save_to_mesh(ring_base_verts, ring_base_faces, '{}/hitdlr_ring_base.obj'.format(path))
            save_to_mesh(ring_proximal_verts, ring_proximal_faces, '{}/hitdlr_ring_proximal.obj'.format(path))
            save_to_mesh(ring_medial_verts, ring_medial_faces, '{}/hitdlr_ring_medial.obj'.format(path))
            save_to_mesh(ring_distal_verts, ring_distal_faces, '{}/hitdlr_ring_distal.obj'.format(path))
            save_to_mesh(little_base_verts, little_base_faces, '{}/hitdlr_little_base.obj'.format(path))
            save_to_mesh(little_proximal_verts, little_proximal_faces, '{}/hitdlr_little_proximal.obj'.format(path))
            save_to_mesh(little_medial_verts, little_medial_faces, '{}/hitdlr_little_medial.obj'.format(path))
            save_to_mesh(little_distal_verts, little_distal_faces, '{}/hitdlr_little_distal.obj'.format(path))
            
            all_verts = np.concatenate([righthand_verts, thumb_base_verts, thumb_proximal_verts, thumb_medial_verts,
                         thumb_distal_verts,fore_base_verts, fore_proximal_verts, fore_medial_verts,
                         fore_distal_verts,middle_base_verts, middle_proximal_verts, middle_medial_verts,
                         middle_distal_verts,ring_base_verts, ring_proximal_verts, ring_medial_verts,
                         ring_distal_verts,little_base_verts, little_proximal_verts, little_medial_verts,
                         little_distal_verts])
            all_faces = np.concatenate([righthand_faces, thumb_base_faces, thumb_proximal_faces, thumb_medial_faces,
                         thumb_distal_faces,fore_base_faces, fore_proximal_faces, fore_medial_faces,
                         fore_distal_faces,middle_base_faces, middle_proximal_faces, middle_medial_faces,
                         middle_distal_faces,ring_base_faces, ring_proximal_faces, ring_medial_faces,
                         ring_distal_faces,little_base_faces, little_proximal_faces, little_medial_faces,
                         little_distal_faces])
            save_to_mesh(all_verts, all_faces, '{}/init_mesh.obj'.format(path))
            hand_mesh = []
            for root, dirs, files in os.walk('{}'.format(path)):
                for filename in files:
                    if filename.endswith('.obj'):
                        filepath = os.path.join(root, filename)
                        mesh = trimesh.load_mesh(filepath)
                        hand_mesh.append(mesh)
            hand_mesh = np.sum(hand_mesh)
        else:
            righthand_mesh = trimesh.Trimesh(righthand_verts, righthand_faces)

            thumb_base_mesh = trimesh.Trimesh(thumb_base_verts, thumb_base_faces)
            thumb_proximal_mesh = trimesh.Trimesh(thumb_proximal_verts, thumb_proximal_faces)
            thumb_medial_mesh = trimesh.Trimesh(thumb_medial_verts, thumb_medial_faces)
            thumb_distal_mesh = trimesh.Trimesh(thumb_distal_verts, thumb_distal_faces)

            fore_base_mesh = trimesh.Trimesh(fore_base_verts, fore_base_faces)
            fore_proximal_mesh = trimesh.Trimesh(fore_proximal_verts, fore_proximal_faces)
            fore_medial_mesh = trimesh.Trimesh(fore_medial_verts, fore_medial_faces)
            fore_distal_mesh = trimesh.Trimesh(fore_distal_verts, fore_distal_faces)

            middle_base_mesh = trimesh.Trimesh(middle_base_verts, middle_base_faces)
            middle_proximal_mesh = trimesh.Trimesh(middle_proximal_verts, middle_proximal_faces)
            middle_medial_mesh = trimesh.Trimesh(middle_medial_verts, middle_medial_faces)
            middle_distal_mesh = trimesh.Trimesh(middle_distal_verts, middle_distal_faces)

            ring_base_mesh = trimesh.Trimesh(ring_base_verts, ring_base_faces)
            ring_proximal_mesh = trimesh.Trimesh(ring_proximal_verts, ring_proximal_faces)
            ring_medial_mesh = trimesh.Trimesh(ring_medial_verts, ring_medial_faces)
            ring_distal_mesh = trimesh.Trimesh(ring_distal_verts, ring_distal_faces)

            little_base_mesh = trimesh.Trimesh(little_base_verts, little_base_faces)
            little_proximal_mesh = trimesh.Trimesh(little_proximal_verts, little_proximal_faces)
            little_medial_mesh = trimesh.Trimesh(little_medial_verts, little_medial_faces)
            little_distal_mesh = trimesh.Trimesh(little_distal_verts, little_distal_faces)
            hand_mesh = [righthand_mesh,
                         thumb_base_mesh, thumb_proximal_mesh, thumb_medial_mesh, thumb_distal_mesh,
                         fore_base_mesh, fore_proximal_mesh, fore_medial_mesh, fore_distal_mesh,
                         middle_base_mesh, middle_proximal_mesh, middle_medial_mesh, middle_distal_mesh,
                         ring_base_mesh, ring_proximal_mesh, ring_medial_mesh, ring_distal_mesh,
                         little_base_mesh, little_proximal_mesh, little_medial_mesh, little_distal_mesh
                         ]
        return hand_mesh

    def get_forward_hand_mesh(self, pose, theta, save_mesh=True, path='./output_mesh'):
        outputs = self.forward(pose, theta)
        vertices_list = [output.squeeze().detach().cpu().numpy() for output in outputs]
        mesh = self.get_hand_mesh(vertices_list, self.gripper_faces, save_mesh=save_mesh, path=path)
        return mesh


if __name__ == "__main__":
    device = 'cuda:0'
    hit = HitdlrLayer(device).to(device)
    # print(hit.gripper_faces)
    # exit()
    pose = torch.from_numpy(np.identity(4)).to(device).reshape(-1, 4, 4).float()
    print(pose.shape)
    theta = np.radians(np.array(grasp_dict_20f['DLR_init']['joint_init']).astype(np.float32))
    theta = torch.from_numpy(theta).to(device).reshape(-1, 20)
    print(theta.shape)
    # theta = torch.zeros((1, 20), dtype=torch.float32).to(device)

    # theta[0, 4] = 1
    # theta[0, 1] = 0.5
    # theta[0, 2] = 0.0
    # theta[0, 3] = 0.0
    mesh = hit.get_forward_hand_mesh(pose, theta, save_mesh=True, path='./output_mesh')

    # T = torch.from_numpy(np.load('./T.npy').astype(np.float32))
    # mesh.apply_transform(T)
    # mesh.show()
    # mesh.apply_transofrmation(T)

    # mesh.show()
    # super_mesh = trimesh.load_mesh('~/super_mesh.stl')
    # super_mesh.visual.face_colors = [0, 255, 0]
    mesh.export('./output_mesh/hand.stl')
    scene = trimesh.Scene([mesh])
    scene.show()
    # (mesh + super_mesh).show()

    # outputs = hit.forward(pose, theta=theta)
    # vertices_list = [output.squeeze().detach().cpu().numpy() for output in outputs]
    # print(vertices[2].shape)
    # print(vertices[3].shape)
    # save_hand(vertices_list, hit.gripper_faces)
    # palm_1_pc = trimesh.PointCloud(vertices[0], colors=[0, 255, 0])
    # palm_2_pc = trimesh.PointCloud(vertices[1], colors=[0, 255, 0])
    #
    # save_to_mesh(vertices, hit.gripper_faces)
    # mesh = trimesh.load_mesh('./hitdlr.obj')
    # mesh.show()
    # exit()
    # base_pc = trimesh.PointCloud(vertices[2].reshape(-1, 3), colors=[0, 255, 0])
    # proximal_pc = trimesh.PointCloud(vertices[3].reshape(-1, 3), colors=[0, 255, 0])
    # medial_pc = trimesh.PointCloud(vertices[4].reshape(-1, 3), colors=[0, 255, 0])
    # distal_pc = trimesh.PointCloud(vertices[5].reshape(-1, 3), colors=[0, 255, 0])
    # # print(vertices[5])
    # # print(distal_pc)
    # scene = trimesh.Scene([palm_1_pc, palm_2_pc, base_pc, proximal_pc, medial_pc, distal_pc])
    # # scene = trimesh.Scene([base_pc, proximal_pc, medial_pc, distal_pc])
    # scene.show()






