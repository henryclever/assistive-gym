import os
import numpy as np
import pybullet as p
from .agent import Agent



import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

sys.path.insert(0, "../smpl/smpl_webuser_py3")
sys.path.insert(0, "../../../smpl/smpl_webuser_py3")

from smpl.smpl_webuser_py3.serialization import load_model



right_arm_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
left_arm_joints = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
right_leg_joints = [28, 29, 30, 31, 32, 33, 34]
left_leg_joints = [35, 36, 37, 38, 39, 40, 41]
head_joints = [20, 21, 22, 23]


class HumanMesh(Agent):
    def __init__(self, dataset_info_dict):
        self.posture = dataset_info_dict['posture']
        self.gender = dataset_info_dict['gender']
        self.data_ct_idx = dataset_info_dict['data_ct'][0]
        self.data_ct_l = dataset_info_dict['data_ct'][1]
        self.data_ct_h = dataset_info_dict['data_ct'][2]
        self.set_num = dataset_info_dict['set_num']
        self.controllable = False
        self.controllable_joint_indices = []
        self.human_type = "mesh"
        self.mesh_type = dataset_info_dict['mesh_type']

        self.pelvis = 0
        self.left_hip = 1
        self.right_hip = 2
        self.stomach = 3
        self.left_knee = 4
        self.right_knee = 5
        self.left_ankle = 7
        self.right_ankle = 8
        self.chest = 9
        self.left_foot = 10
        self.right_foot = 11
        self.neck = 12
        self.left_peck = 13
        self.right_peck = 14
        self.head = 15
        self.left_shoulder = 16
        self.right_shoulder = 17
        self.left_elbow = 18
        self.right_elbow = 19
        self.left_wrist = 20
        self.right_wrist = 21
        self.left_hand = 22
        self.right_hand = 23


    def init(self, human_creation, limits_model, static_human_base, impairment, gender, config, id, np_random, mass=None, radius_scale=1.0, height_scale=1.0, directory = None):
        self.impairment = 'none'
        self.tremors = np.zeros(len(self.controllable_joint_indices))
        # Initialize human

        self.id = id

        pmat_start_x_border = -0.0254 * 3
        pmat_start_y_border = -0.0254 * 2.5
        shift_x = pmat_start_x_border - 0.45775
        shift_y = pmat_start_y_border - 0.98504

        #self.body = human_creation.create_human(static=static_human_base, limit_scale=self.limit_scale, specular_color=[0.1, 0.1, 0.1], gender=self.gender, config=config, mass=mass, radius_scale=radius_scale, height_scale=height_scale)

        if self.mesh_type == 'ground_truth':
            #self.body = p.loadURDF(os.path.join(directory, 'human_mesh', 'human_mesh.urdf'),
            #                       #basePosition=[-0.419, -0.864, 0.3048],
            #                       #basePosition=[-0.45776, -0.98504, 0.3048+0.0254],
            #                       basePosition=[shift_x, shift_y, 0.3048+0.075],
            #                       baseOrientation=p.getQuaternionFromEuler([0.0, 0, 0], physicsClientId=id),
            #                       physicsClientId=id)


            print("loaded human URDF", directory)
            human_visual = p.createVisualShape(shapeType=p.GEOM_MESH,
                                               fileName=os.path.join(directory, 'human_mesh','human.obj'),
                                               rgbaColor=[0.2, 0.2, 1.0, 1], specularColor=[0.2, 0.2, 0.2],
                                               meshScale=[1.0, 1.0, 1.0], physicsClientId=id)
            #
            human_collision = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                               fileName=os.path.join(directory, 'human_mesh','human_vhacd.obj'),
                                               meshScale=[1.0, 1.0, 1.0], flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
                                               physicsClientId=id)
            self.body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=human_collision,
                                          baseVisualShapeIndex=human_visual, basePosition=[shift_x, shift_y, 0.3048+0.075],
                                          useMaximalCoordinates=True, physicsClientId=id)



        elif self.mesh_type == 'estimate':
            self.body = p.loadURDF(os.path.join(directory, 'human_mesh', 'human_mesh_est.urdf'),
                                   #basePosition=[-0.419, -0.864, 0.3048],
                                   basePosition=[0.0, 0.0, 0.0],#[-0.45776, -0.98504, 0.3048],
                                   baseOrientation=p.getQuaternionFromEuler([0.0, 0, 0], physicsClientId=id),
                                   physicsClientId=id)
            print("loaded human estimate URDF")




    def assign_new_pose(self, m, smpl_verts, joint_locs_trans_abs):


        self.legacy_x_shift = -0.286 + 0.0143
        self.legacy_y_shift = -0.286 + 0.0143
        self.pmat_ht = 0.075

        joint_locs_trans_abs = np.array(joint_locs_trans_abs)

        self.joint_locs_trans_abs = []

        for i in range(24):
            to_append = np.array(joint_locs_trans_abs[i]) + np.array([self.legacy_x_shift, self.legacy_y_shift, self.pmat_ht])
            self.joint_locs_trans_abs.append(to_append)

        self.m = m
        self.smpl_verts = smpl_verts



    def get_mesh_pos_orient(self, link, center_of_mass=False, convert_to_realworld=False):
        pos = self.joint_locs_trans_abs[link]
        orient = self.quat_from_dir_cos_angles(np.array([self.m.pose[link], self.m.pose[link+1], self.m.pose[link+2]]))

        return np.array(pos), np.array(orient)

    def get_random_mesh_loc(self, chosen_vert_ind = None):

        faces = np.array(self.m.f)

        if chosen_vert_ind is None:
            # randomly sample a vertex that is weighted by the area of its surrounding triangles
            #we also need to use raycasting to see if it's part of the body that a tool can access from above
            vert_weights = self.get_triangle_area_vert_weight(self.smpl_verts, faces)/6890.
            chosen_vert_fails_raycasting = True
            while chosen_vert_fails_raycasting == True:
                chosen_vert_fails_raycasting = False
                ind = np.where(np.random.multinomial(1, vert_weights))[0][0]
                print("randomly chose vert, idx is:", ind)

                chosen_vert = np.array(self.smpl_verts[ind, :])
                for face_idx in range(np.shape(faces)[0]):
                    vert_inds = faces[face_idx, :]
                    vA = np.array(self.smpl_verts[vert_inds[0], :])
                    vB = np.array(self.smpl_verts[vert_inds[1], :])
                    vC = np.array(self.smpl_verts[vert_inds[2], :])
                    if vA[0] <= chosen_vert[0] and vB[0] <= chosen_vert[0] and vC[0] <= chosen_vert[0]:
                        continue
                    elif vA[0] >= chosen_vert[0] and vB[0] >= chosen_vert[0] and vC[0] >= chosen_vert[0]:
                        continue
                    elif vA[1] <= chosen_vert[1] and vB[1] <= chosen_vert[1] and vC[1] <= chosen_vert[1]:
                        continue
                    elif vA[1] >= chosen_vert[1] and vB[1] >= chosen_vert[1] and vC[1] >= chosen_vert[1]:
                        continue
                    elif np.linalg.norm(chosen_vert - vA) == 0 or np.linalg.norm(chosen_vert - vB) == 0 or np.linalg.norm(chosen_vert - vC) == 0:
                        continue
                    else:

                        d_point_AB = (chosen_vert[0] - vA[0])*(vB[1] - vA[1]) - (chosen_vert[1] - vA[1])*(vB[0] - vA[0])
                        d_C_AB = (vC[0] - vA[0])*(vB[1] - vA[1]) - (vC[1] - vA[1])*(vB[0] - vA[0])

                        d_point_BC = (chosen_vert[0] - vB[0])*(vC[1] - vB[1]) - (chosen_vert[1] - vB[1])*(vC[0] - vB[0])
                        d_A_BC = (vA[0] - vB[0])*(vC[1] - vB[1]) - (vA[1] - vB[1])*(vC[0] - vB[0])

                        d_point_AC = (chosen_vert[0] - vC[0])*(vA[1] - vC[1]) - (chosen_vert[1] - vC[1])*(vA[0] - vC[0])
                        d_B_AC = (vB[0] - vC[0])*(vA[1] - vC[1]) - (vB[1] - vC[1])*(vA[0] - vC[0])

                        if d_point_AB*d_C_AB >=0 and d_point_BC*d_A_BC >= 0 and d_point_AC*d_B_AC >= 0:
                            if np.abs(d_C_AB) - np.abs(d_point_AB) > 0 and np.abs(d_A_BC) - np.abs(d_point_BC) > 0 and np.abs(d_B_AC) - np.abs(d_point_AC) > 0:
                                #print("same side of lines and between")
                                if vA[2] > chosen_vert[2] or vB[2] > chosen_vert[2] or vC[2] > chosen_vert[2]:
                                    #print("RAYTRACING FOUND SURF ABOVE:", chosen_vert, "   probability:", vert_weights[ind] * 6890.)
                                    #print("                OTHERS:", vA, vB, vC)
                                    #print("                ", d_point_AB, d_C_AB)
                                    #print("                ", d_point_BC, d_A_BC)
                                    #print("                ", d_point_AC, d_B_AC)
                                    #print("                       ", d_point_AB * d_C_AB, d_point_BC * d_A_BC, d_point_AC * d_B_AC)
                                    #print("                       ", np.abs(d_C_AB) - np.abs(d_point_AB), np.abs(d_A_BC) - np.abs(d_point_BC), np.abs(d_B_AC) - np.abs(d_point_AC))
                                    chosen_vert_fails_raycasting = True
                                    print("raytracing found an overlying surface. trying again...")
                                    continue
                            else:
                                pass
                                #print("same side of lines but at least one isn't between")
                        else:
                            pass
                            #print("one of the points isn't on same side of line")

        else:
            print("chose existing vert, idx is:", chosen_vert_ind)
            chosen_vert = np.array(self.smpl_verts[chosen_vert_ind, :])


        return chosen_vert



    def get_triangle_area_vert_weight(self, verts, faces):

        # first we need all the triangle areas
        tri_verts = verts[faces, :]
        a = np.linalg.norm(tri_verts[:, 0] - tri_verts[:, 1], axis=1)
        b = np.linalg.norm(tri_verts[:, 1] - tri_verts[:, 2], axis=1)
        c = np.linalg.norm(tri_verts[:, 2] - tri_verts[:, 0], axis=1)
        s = (a + b + c) / 2
        A = np.sqrt(s * (s - a) * (s - b) * (s - c))

        # print np.shape(verts), np.shape(faces), np.shape(A), np.mean(A), 'area'

        A = np.swapaxes(np.stack((A, A, A)), 0, 1)  # repeat the area for each vert in the triangle
        A = A.flatten()
        faces = np.array(faces).flatten()
        i = np.argsort(faces)  # sort the faces and the areas by the face idx
        faces_sorted = faces[i]
        A_sorted = A[i]
        last_face = 0
        area_minilist = []
        area_avg_list = []
        face_sort_list = []  # take the average area for all the trianges surrounding each vert
        for vtx_connect_idx in range(np.shape(faces_sorted)[0]):
            if faces_sorted[vtx_connect_idx] == last_face and vtx_connect_idx != np.shape(faces_sorted)[0] - 1:
                area_minilist.append(A_sorted[vtx_connect_idx])
            elif faces_sorted[vtx_connect_idx] > last_face or vtx_connect_idx == np.shape(faces_sorted)[0] - 1:
                if len(area_minilist) != 0:
                    area_avg_list.append(np.mean(area_minilist))
                else:
                    area_avg_list.append(0)
                face_sort_list.append(last_face)
                area_minilist = []
                last_face += 1
                if faces_sorted[vtx_connect_idx] == last_face:
                    area_minilist.append(A_sorted[vtx_connect_idx])
                elif faces_sorted[vtx_connect_idx] > last_face:
                    num_tack_on = np.copy(faces_sorted[vtx_connect_idx] - last_face)
                    for i in range(num_tack_on):
                        area_avg_list.append(0)
                        face_sort_list.append(last_face)
                        last_face += 1
                        if faces_sorted[vtx_connect_idx] == last_face:
                            area_minilist.append(A_sorted[vtx_connect_idx])

        # print np.mean(area_avg_list), 'area avg'

        area_avg = np.array(area_avg_list)
        area_avg_red = area_avg[
            area_avg > 0]  # find out how many of the areas correspond to verts facing the camera

        # print np.mean(area_avg_red), 'area avg'
        # print np.sum(area_avg_red), np.sum(area_avg)

        norm_area_avg = area_avg / np.sum(area_avg_red)
        norm_area_avg = norm_area_avg * np.shape(area_avg_red)  # multiply by the REDUCED num of verts
        # print norm_area_avg[0:3], np.min(norm_area_avg), np.max(norm_area_avg), np.mean(norm_area_avg), np.sum(norm_area_avg)
        # print norm_area_avg.shape, np.shape(verts_idx_red)

        # print np.shape(verts_idx_red), np.min(verts_idx_red), np.max(verts_idx_red)
        # print np.shape(norm_area_avg), np.min(norm_area_avg), np.max(norm_area_avg)

        #try:
        #    norm_area_avg = norm_area_avg[verts]
        #except:
        #    norm_area_avg = norm_area_avg[verts[:-1]]

        # print norm_area_avg[0:3], np.min(norm_area_avg), np.max(norm_area_avg), np.mean(norm_area_avg), np.sum(norm_area_avg)
        return norm_area_avg





    def quat_from_dir_cos_angles(self,theta):
        angle = np.linalg.norm(theta)
        normalized = theta / angle
        angle = angle * 0.5
        v_cos = np.cos(angle)
        v_sin = np.sin(angle)
        quat = np.array([v_cos, v_sin * normalized[0], v_sin * normalized[1], v_sin * normalized[2]])

        return quat


    def get_mesh_joint_angles(self, controllable_joint_indices):

        mesh_joint_angles = []
        for i in range(72):
            mesh_joint_angles.append(float(self.m.pose[i]))
        print('getting mesh joint angles')
        return mesh_joint_angles

