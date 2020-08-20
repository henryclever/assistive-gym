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

        self.right_pecs = 2
        self.right_shoulder = 5
        self.right_elbow = 7
        self.right_wrist = 9
        self.left_pecs = 12
        self.left_shoulder = 15
        self.left_elbow = 17
        self.left_wrist = 19
        self.neck = 20
        self.head = 23
        self.stomach = 24
        self.waist = 27
        self.right_hip = 30
        self.right_knee = 5
        self.right_ankle = 34
        self.left_hip = 37
        self.left_knee = 38
        self.left_ankle = 41

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




    def assign_new_pose(self, m, joint_locs_trans_abs):

        legacy_x_shift = -0.286 + 0.0143
        legacy_y_shift = -0.286 + 0.0143
        pmat_ht = 0.075


        self.joint_locs_trans_abs = []

        for i in range(24):
            to_append = np.array(joint_locs_trans_abs[i]) + np.array([legacy_x_shift, legacy_y_shift, pmat_ht])
            self.joint_locs_trans_abs.append(to_append)

        self.m = m


    def get_mesh_pos_orient(self, link, center_of_mass=False, convert_to_realworld=False):
        pos = self.joint_locs_trans_abs[link]
        orient = self.quat_from_dir_cos_angles(np.array([self.m.pose[link], self.m.pose[link+1], self.m.pose[link+2]]))

        return np.array(pos), np.array(orient)

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

