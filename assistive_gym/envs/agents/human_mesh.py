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

        pmat_start_x_border = -0.0254*3
        pmat_start_y_border = -0.0254*2.5

        #self.body = human_creation.create_human(static=static_human_base, limit_scale=self.limit_scale, specular_color=[0.1, 0.1, 0.1], gender=self.gender, config=config, mass=mass, radius_scale=radius_scale, height_scale=height_scale)

        if self.mesh_type == 'ground_truth':
            self.body = p.loadURDF(os.path.join(directory, 'human_mesh', 'human_mesh.urdf'),
                                   #basePosition=[-0.419, -0.864, 0.3048],
                                   #basePosition=[-0.45776, -0.98504, 0.3048+0.0254],
                                   basePosition=[pmat_start_x_border, pmat_start_y_border, 0.3048+0.075],
                                   baseOrientation=p.getQuaternionFromEuler([0.0, 0, 0], physicsClientId=id),
                                   physicsClientId=id)
            print("loaded human URDF")


        elif self.mesh_type == 'estimate':
            self.body = p.loadURDF(os.path.join(directory, 'human_mesh', 'human_mesh_est.urdf'),
                                   #basePosition=[-0.419, -0.864, 0.3048],
                                   basePosition=[0.0, 0.0, 0.0],#[-0.45776, -0.98504, 0.3048],
                                   baseOrientation=p.getQuaternionFromEuler([0.0, 0, 0], physicsClientId=id),
                                   physicsClientId=id)
            print("loaded human estimate URDF")



    def load_smpl_model(self, joint_locs_trans_abs):

        #load SMPL model built for python3
        #load resting pose data
        resting_post_filename = "resting_pose_roll0_"+self.gender[0]+"_lay_set"+str(self.set_num)+"_"+str(self.data_ct_l)+"_of_"+str(self.data_ct_h)+"_none_stiff.npy"
        resting_pose_data = np.load(resting_post_filename, allow_pickle = True, encoding='latin1')
        print (np.shape(resting_pose_data), np.shape(resting_pose_data[0, 0]), np.shape(resting_pose_data[0, 1]), np.shape(resting_pose_data[0, 2]), np.shape(resting_pose_data[0, 3]))

        model_path = '../smpl/models/basicModel_' + self.gender[0] + '_lbs_10_207_0_v1.0.0.pkl'
        m = load_model(model_path)

        # first get the offsets
        DART_TO_FLEX_CONV = 2.58872
        PERSON_SCALE = 50.0
        mTransX = m.r[0, 0] * DART_TO_FLEX_CONV / (PERSON_SCALE * 0.1)  # X appears to move sideways
        mTransY = m.r[0, 1] * DART_TO_FLEX_CONV / (PERSON_SCALE * 0.1)  # Y trans appears to move up in air
        mTransZ = m.r[0, 2] * DART_TO_FLEX_CONV / (PERSON_SCALE * 0.1)  # Z trans appears to move forward
        mTrans = [mTransX, mTransY, mTransZ]


        capsule_angles = resting_pose_data[self.data_ct_idx, 0].tolist()
        root_joint_pos_list = resting_pose_data[self.data_ct_idx, 1]
        body_shape_list = resting_pose_data[self.data_ct_idx, 2]

        for shape_param in range(10):
            m.betas[shape_param] = float(body_shape_list[shape_param])

        m.pose[0:3] = capsule_angles[0:3]
        m.pose[3:6] = capsule_angles[6:9]
        m.pose[6:9] = capsule_angles[9:12]
        m.pose[9:12] = capsule_angles[12:15]
        m.pose[12] = capsule_angles[15]
        m.pose[15] = capsule_angles[16]
        m.pose[18:21] = capsule_angles[17:20]
        m.pose[21:24] = capsule_angles[20:23]
        m.pose[24:27] = capsule_angles[23:26]
        m.pose[27:30] = capsule_angles[26:29]
        m.pose[36:39] = capsule_angles[29:32] # neck
        m.pose[39:42] = capsule_angles[32:35]
        m.pose[42:45] = capsule_angles[35:38]
        m.pose[45:48] = capsule_angles[38:41]  # head
        m.pose[48:51] = capsule_angles[41:44]
        m.pose[51:54] = capsule_angles[44:47]
        m.pose[55] = capsule_angles[47]
        m.pose[58] = capsule_angles[48]
        m.pose[60:63] = capsule_angles[49:52]
        m.pose[63:66] = capsule_angles[52:55]

        # euler angles
        # angle1 should flip about body
        # angle2 should flip upside down, i.e. about head to toe axis
        # angle3 should flip head to toe, i.e. about gravity axis

        # get the starting height for a flat bed.
        mJ_transformed = np.asarray(m.J_transformed).astype(float)
        mJ = np.asarray(m.J).astype(float)

        #print(mJ_transformed[0, :])
        #print(mJ)
        #mJ_transformed[0, :] += np.array(root_joint_pos_list)
        #mJ_transformed[0, :] += mTrans
        print(mTrans, 'mtrans')
        print(root_joint_pos_list)
        print(mJ_transformed[0, :])

        #mTrans += np.array([0.0, 0.0, -0.0])




    def assign_new_pose(self, m, joint_locs_trans_abs):

        legacy_x_shift = -0.286 + 0.0143
        legacy_y_shift = -0.286 + 0.0143

        bed_leg_ht = 0.3048
        mattress_ht = 0.2032
        pmat_ht = 0.075


        self.joint_locs_trans_abs = []
        self.joint_locs_trans_abs2 = []

        for i in range(24):
            #self.joint_locs_trans_abs.append(list(mJ_transformed[i, :] +mTrans+np.array(root_joint_pos_list)+np.array([0.45776, 0.98504, 0.3048])))# mJ_transformed[0, :]))#))# + mTrans))

            to_append = np.array(joint_locs_trans_abs[i]) + np.array([legacy_x_shift, legacy_y_shift, bed_leg_ht + mattress_ht + pmat_ht])
            self.joint_locs_trans_abs.append(to_append[0])

            #print('orig:', self.joint_locs_trans_abs[-1], '   proposed:',self.joint_locs_trans_abs2[-1])

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

