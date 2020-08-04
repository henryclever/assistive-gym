import os
import numpy as np
import pybullet as p
from .agent import Agent
from smpl_webuser3.serialization import load_model


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
        self.right_knee = 31
        self.right_ankle = 34
        self.left_hip = 37
        self.left_knee = 38
        self.left_ankle = 41

    def init(self, human_creation, limits_model, static_human_base, impairment, gender, config, id, np_random, mass=None, radius_scale=1.0, height_scale=1.0, directory = None):
        self.impairment = 'none'
        self.tremors = np.zeros(len(self.controllable_joint_indices))
        # Initialize human

        self.id = id

        #self.body = human_creation.create_human(static=static_human_base, limit_scale=self.limit_scale, specular_color=[0.1, 0.1, 0.1], gender=self.gender, config=config, mass=mass, radius_scale=radius_scale, height_scale=height_scale)


        self.body = p.loadURDF(os.path.join(directory, 'human_mesh', 'human_mesh.urdf'),
                               basePosition=[-0.419, -0.864, 0.3048],
                               baseOrientation=p.getQuaternionFromEuler([0.0, 0, 0], physicsClientId=id),
                               physicsClientId=id)
        print("loaded human URDF")


        #load SMPL model built for python3

        #load resting pose data
        resting_post_filename = "/home/henry/data/resting_poses/"+self.posture+"/resting_pose_roll0_"+self.gender[0]+"_lay_set"+str(self.set_num)+"_"+str(self.data_ct_l)+"_of_"+str(self.data_ct_h)+"_none_stiff.npy"
        resting_pose_data = np.load(resting_post_filename, allow_pickle = True, encoding='latin1')
        print (np.shape(resting_pose_data), np.shape(resting_pose_data[0, 0]), np.shape(resting_pose_data[0, 1]), np.shape(resting_pose_data[0, 2]), np.shape(resting_pose_data[0, 3]))

        model_path = '/home/henry/git/SMPL_python_v.1.0.0/smpl/models/basicModel_' + self.gender[0] + '_lbs_10_207_0_v1.0.0.pkl'
        m = load_model(model_path)

        PERSON_SCALE = 50.0

        # first get the offsets
        DART_TO_FLEX_CONV = 2.5872
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

        human_rotation = 0.0#capsule_angles[2] #do not use this! It's already embedded in the pose
        human_shiftSIDE = root_joint_pos_list[0]*2.58872
        human_shiftUD = root_joint_pos_list[1]*2.58872

        #print human_shiftUD, capsule_angles[3:6]

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


        mTransX, mTransY, mTransZ = mTrans

        ROOT_ROT_X = 0
        ROOT_ROT_Y = 0
        ROOT_ROT_Z = human_rotation


        # print m.pose
        m.pose[0] = ROOT_ROT_X
        m.pose[1] = ROOT_ROT_Y
        m.pose[2] = ROOT_ROT_Z


        # euler angles
        # angle1 should flip about body
        # angle2 should flip upside down, i.e. about head to toe axis
        # angle3 should flip head to toe, i.e. about gravity axis

        # get the starting height for a flat bed.
        mJ_transformed = np.asarray(m.J_transformed)
        self.joint_locs_trans_abs = []
        for i in range(24):
            self.joint_locs_trans_abs.append(list(mJ_transformed[i, :] - mJ_transformed[0, :]))

        self.m = m




        #super(HumanMesh, self).init(self.body, id, np_random, self.controllable_joint_indices)


    def update_human_mesh(self):

        mesh_data_folder = "/home/henry/data/resting_meshes/"+self.posture+"/roll0_"+self.gender[0]+"_lay_set"+str(self.set_num)+"_"+str(self.data_ct_l)+"_of_"+str(self.data_ct_h)+"_none_stiff/"
        hv = np.load(mesh_data_folder+"hv.npy")#, allow_pickle = True, encoding='latin1')
        hf = np.load(mesh_data_folder+"hf.npy")#, allow_pickle = True, encoding='latin1')

        for sample_idx in range(1501, 1502):# (0, np.shape(mv)[0]):
            human_verts = np.array(hv[sample_idx, :, :])/2.58872 #np.load(folder + "human_mesh_verts_py.npy")/2.58872
            human_verts = np.concatenate((human_verts[:, 2:3],human_verts[:, 0:1],human_verts[:, 1:2]), axis = 1)

            human_faces = np.array(hf[sample_idx, :, :]) #np.load(folder + "human_mesh_faces_py.npy")
            human_faces = np.concatenate((np.array([[0, 1, 2], [0, 4, 1], [0, 5, 4], [0, 2, 132], [0, 235, 5], [0, 132, 235] ]), human_faces), axis = 0)
            human_vf = [human_verts, human_faces]

        outmesh_human_path = "/home/henry/git/assistive-gym/assistive_gym/envs/assets/human_mesh/human.obj"
        with open(outmesh_human_path, 'w') as fp:
            for v_idx in range(human_verts.shape[0]):
                fp.write('v %f %f %f\n' % (human_verts[v_idx, 0], human_verts[v_idx, 1], human_verts[v_idx, 2]))

            for f_idx in range(human_faces.shape[0]):
                fp.write('f %d %d %d\n' % (human_faces[f_idx, 0]+1, human_faces[f_idx, 1]+1, human_faces[f_idx, 2]+1))

    def get_mesh_pos_orient(self, link, center_of_mass=False, convert_to_realworld=False):
        pos = self.joint_locs_trans_abs[link]
        orient = np.array([self.m.pose[link], self.m.pose[link+1], self.m.pose[link+2]])

        return np.array(pos), np.array(orient)


class Human(Agent):
    def __init__(self, controllable_joint_indices, controllable=False):
        super(Human, self).__init__()
        self.human_type = "kinematic"
        self.controllable_joint_indices = controllable_joint_indices
        self.controllable = controllable
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
        self.right_knee = 31
        self.right_ankle = 34
        self.left_hip = 37
        self.left_knee = 38
        self.left_ankle = 41

        self.j_right_pecs_x, self.j_right_pecs_y, self.j_right_pecs_z = 0, 1, 2
        self.j_right_shoulder_x, self.j_right_shoulder_y, self.j_right_shoulder_z = 3, 4, 5
        self.j_right_elbow = 6
        self.j_right_forearm = 7
        self.j_right_wrist_x, self.j_right_wrist_y = 8, 9
        self.j_left_pecs_x, self.j_left_pecs_y, self.j_left_pecs_z = 10, 11, 12
        self.j_left_shoulder_x, self.j_left_shoulder_y, self.j_left_shoulder_z = 13, 14, 15
        self.j_left_elbow = 16
        self.j_left_forearm = 17
        self.j_left_wrist_x, self.j_left_wrist_y = 18, 19
        self.j_neck = 20
        self.j_head_x, self.j_head_y, self.j_head_z = 21, 22, 23
        self.j_waist_x, self.j_waist_y, self.j_waist_z = 25, 26, 27
        self.j_right_hip_x, self.j_right_hip_y, self.j_right_hip_z = 28, 29, 30
        self.j_right_knee = 31
        self.j_right_ankle_x, self.j_right_ankle_y, self.j_right_ankle_z = 32, 33, 34
        self.j_left_hip_x, self.j_left_hip_y, self.j_left_hip_z = 35, 36, 37
        self.j_left_knee = 38
        self.j_left_ankle_x, self.j_left_ankle_y, self.j_left_ankle_z = 39, 40, 41

        self.impairment = 'random'
        self.limit_scale = 1.0
        self.strength = 1.0
        self.tremors = np.zeros(10)
        self.target_joint_angles = None
        self.hand_radius = 0.0
        self.elbow_radius = 0.0
        self.shoulder_radius = 0.0

    def init(self, human_creation, limits_model, static_human_base, impairment, gender, config, id, np_random, mass=None, radius_scale=1.0, height_scale=1.0, directory = None):
        self.limits_model = limits_model
        self.arm_previous_valid_pose = {True: None, False: None}
        # Choose gender
        if gender not in ['male', 'female']:
            gender = np_random.choice(['male', 'female'])
        self.gender = gender
        # Specify human impairments
        if impairment == 'random':
            impairment = np_random.choice(['none', 'limits', 'weakness', 'tremor'])
        elif impairment == 'no_tremor':
            impairment = np_random.choice(['none', 'limits', 'weakness'])
        self.impairment = impairment
        self.limit_scale = 1.0 if impairment != 'limits' else np_random.uniform(0.5, 1.0)
        self.strength = 1.0 if impairment != 'weakness' else np_random.uniform(0.25, 1.0)
        if self.impairment != 'tremor':
            self.tremors = np.zeros(len(self.controllable_joint_indices))
        elif self.head in self.controllable_joint_indices:
            self.tremors = np_random.uniform(np.deg2rad(-20), np.deg2rad(20), size=len(self.controllable_joint_indices))
        else:
            self.tremors = np_random.uniform(np.deg2rad(-10), np.deg2rad(10), size=len(self.controllable_joint_indices))
        # Initialize human
        self.body = human_creation.create_human(static=static_human_base, limit_scale=self.limit_scale, specular_color=[0.1, 0.1, 0.1], gender=self.gender, config=config, mass=mass, radius_scale=radius_scale, height_scale=height_scale)
        self.hand_radius = human_creation.hand_radius
        self.elbow_radius = human_creation.elbow_radius
        self.shoulder_radius = human_creation.shoulder_radius

        super(Human, self).init(self.body, id, np_random, self.controllable_joint_indices)

        # By default, initialize the person in the wheelchair
        self.set_base_pos_orient([0, 0.03, 0.89 if self.gender == 'male' else 0.86], [0, 0, 0, 1])

    def setup_joints(self, joints_positions, use_static_joints=True, reactive_force=None, reactive_gain=0.05):
        # Set static joints
        joint_angles = self.get_joint_angles_dict(self.all_joint_indices)
        for j in self.all_joint_indices:
            if use_static_joints and (j not in self.controllable_joint_indices or (self.impairment != 'tremor' and reactive_force is None)):
                # Make all non controllable joints on the person static by setting mass of each link (joint) to 0
                p.changeDynamics(self.body, j, mass=0, physicsClientId=self.id)
                # Set velocities to 0
                self.set_joint_angles([j], [joint_angles[j]])

        # Set starting joint positions
        self.set_joint_angles([j for j, _ in joints_positions], [np.deg2rad(j_angle) for _, j_angle in joints_positions])

        # By default, all joints have motors enabled by default that prevent free motion. Disable these motors.
        for j in self.all_joint_indices:
            p.setJointMotorControl2(self.body, jointIndex=j, controlMode=p.VELOCITY_CONTROL, force=0, physicsClientId=self.id)

        self.enforce_joint_limits()

        self.target_joint_angles = self.get_joint_angles(self.controllable_joint_indices)
        if reactive_force is not None:
            # NOTE: This runs a Position / Velocity PD controller for each joint motor on the human
            forces = [reactive_force * self.strength] * len(self.target_joint_angles)
            self.control(self.controllable_joint_indices, self.target_joint_angles, reactive_gain, forces)

    def enforce_realistic_joint_limits(self):
        # Only enforce limits for the human arm that is moveable (if either arm is even moveable)
        if (self.j_right_shoulder_x not in self.controllable_joint_indices) and (self.j_left_shoulder_x not in self.controllable_joint_indices):
            return
        right = self.j_right_shoulder_x in self.controllable_joint_indices
        indices = [self.j_right_shoulder_x, self.j_right_shoulder_y, self.j_right_shoulder_z, self.j_right_elbow] if right else [self.j_right_shoulder_x, self.j_right_shoulder_y, self.j_right_shoulder_z, self.j_right_elbow]
        tz, tx, ty, qe = self.get_joint_angles(indices)
        # Transform joint angles to match those from the Matlab data
        tz2 = (((-1 if right else 1)*tz) + 2*np.pi) % (2*np.pi)
        tx2 = (tx + 2*np.pi) % (2*np.pi)
        ty2 = (-1 if right else 1)*ty
        qe2 = (-qe + 2*np.pi) % (2*np.pi)
        result = self.limits_model.predict_classes(np.array([[tz2, tx2, ty2, qe2]]))
        if result == 1:
            # This is a valid pose for the person
            self.arm_previous_valid_pose[right] = [tz, tx, ty, qe]
        elif result == 0 and self.right_arm_previous_valid_pose is not None:
            # The person is in an invalid pose. Move joint angles back to the most recent valid pose.
            self.set_joint_angles(indices, self.arm_previous_valid_pose[right])

