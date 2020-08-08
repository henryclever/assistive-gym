import os
from gym import spaces
import numpy as np
import pybullet as p
import time

from .env import AssistiveEnv
from .agents import furniture
from .agents.furniture import Furniture

class BedBathingEnv(AssistiveEnv):
    def __init__(self, robot, human, bed_type):
        self.step_num = 0
        self.bed_type = bed_type
        self.time_last = 0.0
        super(BedBathingEnv, self).__init__(robot=robot, human=human, bed_type='default', task='bed_bathing', obs_robot_len=(17 + len(robot.controllable_joint_indices)), obs_human_len=(18 + len(human.controllable_joint_indices)))

    def step(self, action):
        self.step_num += 1

        print('taking step', self.step_num, '  time: ', time.time()-self.time_last)
        self.time_last = time.time()
        self.take_step(action, gains=self.config('robot_gains'), forces=self.config('robot_forces'))

        obs = self._get_obs()

        # Get human preferences
        end_effector_velocity = np.linalg.norm(self.robot.get_velocity(self.robot.left_end_effector))
        preferences_score = self.human_preferences(end_effector_velocity=end_effector_velocity, total_force_on_human=self.total_force_on_human, tool_force_at_target=self.tool_force_on_human)

        reward_distance = -min(self.tool.get_closest_points(self.human, distance=5.0)[-1])
        reward_action = -np.linalg.norm(action) # Penalize actions
        reward_new_contact_points = self.new_contact_points # Reward new contact points on a person

        reward = self.config('distance_weight')*reward_distance + self.config('action_weight')*reward_action + self.config('wiping_reward_weight')*reward_new_contact_points + preferences_score

        if self.gui and self.tool_force_on_human > 0:
            print('Task success:', self.task_success, 'Force at tool on human:', self.tool_force_on_human, reward_new_contact_points)

        info = {'total_force_on_human': self.total_force_on_human, 'task_success': int(self.task_success >= (self.total_target_count*self.config('task_success_threshold'))), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        done = False

        print(self.tool.get_pos_orient(1)[0])

        return obs, reward, done, info

    def get_total_force(self):
        total_force_on_human = np.sum(self.robot.get_contact_points(self.human)[-1])
        tool_force = np.sum(self.tool.get_contact_points()[-1])
        tool_force_on_human = 0
        new_contact_points = 0
        for linkA, linkB, posA, posB, force in zip(*self.tool.get_contact_points(self.human)):
            total_force_on_human += force
            if linkA in [1]:
                tool_force_on_human += force
                # Only consider contact with human upperarm, forearm, hand
                if linkB < 0 or linkB > len(self.human.all_joint_indices):
                    continue

                indices_to_delete = []
                for i, (target_pos_world, target) in enumerate(zip(self.targets_pos_upperarm_world, self.targets_upperarm)):
                    if np.linalg.norm(posB - target_pos_world) < 0.025:
                        # The robot made contact with a point on the person's arm
                        new_contact_points += 1
                        self.task_success += 1
                        target.set_base_pos_orient(self.np_random.uniform(1000, 2000, size=3), [0, 0, 0, 1])
                        indices_to_delete.append(i)
                self.targets_pos_on_upperarm = [t for i, t in enumerate(self.targets_pos_on_upperarm) if i not in indices_to_delete]
                self.targets_upperarm = [t for i, t in enumerate(self.targets_upperarm) if i not in indices_to_delete]
                self.targets_pos_upperarm_world = [t for i, t in enumerate(self.targets_pos_upperarm_world) if i not in indices_to_delete]

                indices_to_delete = []
                for i, (target_pos_world, target) in enumerate(zip(self.targets_pos_forearm_world, self.targets_forearm)):
                    if np.linalg.norm(posB - target_pos_world) < 0.025:
                        # The robot made contact with a point on the person's arm
                        new_contact_points += 1
                        self.task_success += 1
                        target.set_base_pos_orient(self.np_random.uniform(1000, 2000, size=3), [0, 0, 0, 1])
                        indices_to_delete.append(i)
                self.targets_pos_on_forearm = [t for i, t in enumerate(self.targets_pos_on_forearm) if i not in indices_to_delete]
                self.targets_forearm = [t for i, t in enumerate(self.targets_forearm) if i not in indices_to_delete]
                self.targets_pos_forearm_world = [t for i, t in enumerate(self.targets_pos_forearm_world) if i not in indices_to_delete]

        return tool_force, tool_force_on_human, total_force_on_human, new_contact_points

    def _get_obs(self, agent=None):
        tool_pos, tool_orient = self.tool.get_pos_orient(1)
        tool_pos_real, tool_orient_real = self.robot.convert_to_realworld(tool_pos, tool_orient)
        robot_joint_angles = self.robot.get_joint_angles(self.robot.controllable_joint_indices)

        if self.human.human_type == 'kinematic':
            shoulder_pos = self.human.get_pos_orient(self.human.right_shoulder)[0]
            elbow_pos = self.human.get_pos_orient(self.human.right_elbow)[0]
            wrist_pos = self.human.get_pos_orient(self.human.right_wrist)[0]
        elif self.human.human_type == 'mesh':
            shoulder_pos = self.human.get_mesh_pos_orient(self.human.right_shoulder)[0]
            elbow_pos = self.human.get_mesh_pos_orient(self.human.right_elbow)[0]
            wrist_pos = self.human.get_mesh_pos_orient(self.human.right_wrist)[0]



        shoulder_pos_real, _ = self.robot.convert_to_realworld(shoulder_pos)
        elbow_pos_real, _ = self.robot.convert_to_realworld(elbow_pos)
        wrist_pos_real, _ = self.robot.convert_to_realworld(wrist_pos)
        self.tool_force, self.tool_force_on_human, self.total_force_on_human, self.new_contact_points = self.get_total_force()
        if self.human.controllable:
            human_joint_angles = self.human.get_joint_angles(self.human.controllable_joint_indices)
            tool_pos_human, tool_orient_human = self.human.convert_to_realworld(tool_pos, tool_orient)
            shoulder_pos_human, _ = self.human.convert_to_realworld(shoulder_pos)
            elbow_pos_human, _ = self.human.convert_to_realworld(elbow_pos)
            wrist_pos_human, _ = self.human.convert_to_realworld(wrist_pos)

        robot_obs = np.concatenate([tool_pos_real, tool_orient_real, robot_joint_angles, shoulder_pos_real, elbow_pos_real, wrist_pos_real, [self.tool_force]]).ravel()
        if self.human.controllable:
            human_obs = np.concatenate([tool_pos_human, tool_orient_human, human_joint_angles, shoulder_pos_human, elbow_pos_human, wrist_pos_human, [self.total_force_on_human, self.tool_force_on_human]]).ravel()
        else:
            human_obs = []

        if agent == 'robot':
            return robot_obs
        elif agent == 'human':
            return human_obs
        return np.concatenate([robot_obs, human_obs]).ravel()


    def update_mattress_mesh(self):


        mesh_data_folder = "/home/henry/data/resting_meshes/general_supine/roll0_f_lay_set14_2000_of_2109_none_stiff/"
        mv = np.load(mesh_data_folder+"mv.npy")#, allow_pickle = True, encoding='latin1')
        mf = np.load(mesh_data_folder+"mf.npy")#, allow_pickle = True, encoding='latin1')
        bv = np.load(mesh_data_folder+"bv.npy", allow_pickle = True, encoding='latin1')
        bf = np.load(mesh_data_folder+"bf.npy", allow_pickle = True, encoding='latin1')

        for sample_idx in range(1501, 1502):# (0, np.shape(mv)[0]):
            mattress_verts = np.array(mv[sample_idx, :, :]) / 2.58872  # np.load(folder + "mattress_verts_py.npy")/2.58872
            mattress_verts = np.concatenate((mattress_verts[:, 2:3], mattress_verts[:, 0:1], mattress_verts[:, 1:2]), axis=1)

            mattress_faces = np.array(mf[sample_idx, :, :])  # np.load(folder + "mattress_faces_py.npy")
            mattress_faces = np.concatenate((np.array([[0, 6054, 6055]]), mattress_faces), axis=0)
            #mattress_vf = [mattress_verts, mattress_faces]





            mat_blanket_verts = np.array(bv[sample_idx]) / 2.58872  # np.load(folder + "mat_blanket_verts_py.npy")/2.58872
            mat_blanket_verts = np.concatenate((mat_blanket_verts[:, 2:3], mat_blanket_verts[:, 0:1], mat_blanket_verts[:, 1:2]), axis=1)
            mat_blanket_faces = np.array(bf[sample_idx])  # np.load(folder + "mat_blanket_faces_py.npy")

            pmat_faces = []
            for i in range(np.shape(mat_blanket_faces)[0]):
                if np.max(mat_blanket_faces[i, :]) < 2244:  # 2300:# < 10000:
                    pmat_faces.append([mat_blanket_faces[i, 0], mat_blanket_faces[i, 2], mat_blanket_faces[i, 1]])

                # elif np.max(mat_blanket_faces[i, :]) > 3000 and np.max(mat_blanket_faces[i, :])< 3100:  #
                #    stagger = 0
                #    pmat_faces.append([mat_blanket_faces[i, 0]+stagger, mat_blanket_faces[i, 2]+stagger, mat_blanket_faces[i, 1]+stagger])

            pmat_verts = np.copy(mat_blanket_verts)[0:6000]
            pmat_faces = np.array(pmat_faces)
            pmat_faces = np.concatenate((np.array([[0, 69, 1], [0, 68, 69]]), pmat_faces), axis=0)

            pmat_vf = [pmat_verts, pmat_faces]





        outmesh_mattress_path = "/home/henry/git/assistive-gym/assistive_gym/envs/assets/bed_mesh/bed_mattress.obj"
        with open(outmesh_mattress_path, 'w') as fp:
            for v_idx in range(mattress_verts.shape[0]):
                fp.write('v %f %f %f\n' % (mattress_verts[v_idx, 0], mattress_verts[v_idx, 1], mattress_verts[v_idx, 2]))

            for f_idx in range(mattress_faces.shape[0]):
                fp.write('f %d %d %d\n' % (mattress_faces[f_idx, 0]+1, mattress_faces[f_idx, 1]+1, mattress_faces[f_idx, 2]+1))


        outmesh_pmat_path = "/home/henry/git/assistive-gym/assistive_gym/envs/assets/bed_mesh/bed_pmat.obj"
        with open(outmesh_pmat_path, 'w') as fp:
            for v_idx in range(pmat_verts.shape[0]):
                fp.write('v %f %f %f\n' % (pmat_verts[v_idx, 0], pmat_verts[v_idx, 1], pmat_verts[v_idx, 2]))

            for f_idx in range(pmat_faces.shape[0]):
                fp.write('f %d %d %d\n' % (pmat_faces[f_idx, 0]+1, pmat_faces[f_idx, 1]+1, pmat_faces[f_idx, 2]+1))



    def reset(self):
        super(BedBathingEnv, self).reset()




        if self.bed_type == 'pressuresim':
            self.update_mattress_mesh()
            self.human.update_human_mesh()
            self.build_assistive_env(furniture_type='bed_mesh', fixed_human_base=False)
        else:
            self.build_assistive_env(furniture_type='bed', fixed_human_base=False)



        self.furniture.set_friction(self.furniture.base, friction=5)

        # Setup human in the air and let them settle into a resting pose on the bed
        if self.human.human_type == 'kinematic':
            joints_positions = [(self.human.j_right_shoulder_x, 30)]
            self.human.setup_joints(joints_positions, use_static_joints=False, reactive_force=None)
            self.human.set_base_pos_orient([-0.15, 0.2, 0.95], [-np.pi/2.0, 0, 0])

        p.setGravity(0, 0, -1, physicsClientId=self.id)

        if self.human.human_type == 'kinematic':
            # Add small variation in human joint positions
            motor_indices, motor_positions, motor_velocities, motor_torques = self.human.get_motor_joint_states()
            self.human.set_joint_angles(motor_indices, self.np_random.uniform(-0.1, 0.1, size=len(motor_indices)))

        # Let the person settle on the bed
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.id)



        if self.human.human_type == 'kinematic':
            # Lock human joints and set velocities to 0
            joints_positions = []
            self.human.setup_joints(joints_positions, use_static_joints=True, reactive_force=None, reactive_gain=0.01)
            self.human.set_mass(self.human.base, mass=0)
            self.human.set_base_velocity(linear_velocity=[0, 0, 0], angular_velocity=[0, 0, 0])


        if self.human.human_type == 'kinematic':
            shoulder_pos = self.human.get_pos_orient(self.human.right_shoulder)[0]
            elbow_pos = self.human.get_pos_orient(self.human.right_elbow)[0]
            wrist_pos = self.human.get_pos_orient(self.human.right_wrist)[0]
        elif self.human.human_type == 'mesh':
            shoulder_pos = self.human.get_mesh_pos_orient(self.human.right_shoulder)[0]
            elbow_pos = self.human.get_mesh_pos_orient(self.human.right_elbow)[0]
            wrist_pos = self.human.get_mesh_pos_orient(self.human.right_wrist)[0]


        target_ee_pos = np.array([-0.6, 0.2, 1]) + self.np_random.uniform(-0.05, 0.05, size=3)

        print('target ee orient', self.task, id)


        target_ee_orient = np.array(p.getQuaternionFromEuler(np.array(self.robot.toc_ee_orient_rpy[self.task]), physicsClientId=self.id))
        # Use TOC with JLWKI to find an optimal base position for the robot near the person
        base_position, _, _ = self.robot.position_robot_toc(self.task, 'left', [(target_ee_pos, target_ee_orient)], [(shoulder_pos, None), (elbow_pos, None), (wrist_pos, None)], self.human, step_sim=True, check_env_collisions=False)
        if self.robot.wheelchair_mounted:
            # Load a nightstand in the environment for the jaco arm
            self.nightstand = Furniture()
            self.nightstand.init('nightstand', self.directory, self.id, self.np_random)
            self.nightstand.set_base_pos_orient(np.array([-0.9, 0.7, 0]) + base_position, [np.pi/2.0, 0, 0])
        # Open gripper to hold the tool
        self.robot.set_gripper_open_position(self.robot.left_gripper_indices, self.robot.gripper_pos[self.task], set_instantly=True)
        # Initialize the tool in the robot's gripper
        self.tool.init(self.robot, self.task, self.directory, self.id, self.np_random, right=False, mesh_scale=[1]*3)

        self.generate_targets()

        p.setGravity(0, 0, -9.81, physicsClientId=self.id)
        self.robot.set_gravity(0, 0, 0)
        self.human.set_gravity(0, 0, -1)
        self.tool.set_gravity(0, 0, 0)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        self.init_env_variables()
        return self._get_obs()

    def generate_targets(self):
        self.target_indices_to_ignore = []
        if self.human.gender == 'male':
            self.upperarm, self.upperarm_length, self.upperarm_radius = self.human.right_shoulder, 0.279, 0.043
            self.forearm, self.forearm_length, self.forearm_radius = self.human.right_elbow, 0.257, 0.033
        else:
            self.upperarm, self.upperarm_length, self.upperarm_radius = self.human.right_shoulder, 0.264, 0.0355
            self.forearm, self.forearm_length, self.forearm_radius = self.human.right_elbow, 0.234, 0.027

        self.targets_pos_on_upperarm = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -self.upperarm_length]), radius=self.upperarm_radius, distance_between_points=0.03)
        self.targets_pos_on_forearm = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -self.forearm_length]), radius=self.forearm_radius, distance_between_points=0.03)

        print(np.shape(self.targets_pos_on_forearm))
        print(np.shape(self.targets_pos_on_upperarm))

        for idx in range(np.shape(self.targets_pos_on_forearm)[0]):
            self.targets_pos_on_forearm[idx][0] -= 0.5
            self.targets_pos_on_forearm[idx][1] += 0.5
            #self.targets_pos_on_forearm[idx][2] -= 0.5

        for idx in range(np.shape(self.targets_pos_on_upperarm)[0]):
            self.targets_pos_on_upperarm[idx][0] = 0.0
            self.targets_pos_on_upperarm[idx][1] = 0.0
            self.targets_pos_on_upperarm[idx][2] = 0.0

        for i in range(24): #right now this is hacked to show markers at every joint rather than around the upper arm
            pos, _ = self.human.get_mesh_pos_orient(i)
            self.targets_pos_on_upperarm[i][0] = pos[0]
            self.targets_pos_on_upperarm[i][1] = pos[1]
            self.targets_pos_on_upperarm[i][2] = pos[2]


        self.targets_upperarm = self.create_spheres(radius=0.05, mass=0.0, batch_positions=[[0, 0, 0]]*len(self.targets_pos_on_upperarm), visual=True, collision=False, rgba=[1, 0.2, 0.2, 1])
        self.targets_forearm = self.create_spheres(radius=0.01, mass=0.0, batch_positions=[[0, 0, 0]]*len(self.targets_pos_on_forearm), visual=True, collision=False, rgba=[0, 1, 1, 0])

        #print(self.targets_pos_on_forearm)
        self.total_target_count = len(self.targets_pos_on_upperarm) + len(self.targets_pos_on_forearm)
        self.update_targets()

    def update_targets(self):


        if self.human.human_type == 'kinematic':
            upperarm_pos, upperarm_orient = self.human.get_pos_orient(self.upperarm)
        elif self.human.human_type == 'mesh':
            #upperarm_pos, upperarm_orient = self.human.get_mesh_pos_orient(self.upperarm) ##### relative to upper arm! need to get this right.
            upperarm_pos, upperarm_orient = self.human.get_mesh_pos_orient(0) ##### relative to upper arm! need to get this right.
            #print(upperarm_pos)

        self.targets_pos_upperarm_world = []
        for target_pos_on_arm, target in zip(self.targets_pos_on_upperarm, self.targets_upperarm):
            #target_pos = np.array(p.multiplyTransforms(upperarm_pos, upperarm_orient, target_pos_on_arm, [0, 0, 0, 1], physicsClientId=self.id)[0])
            target_pos = np.array(p.multiplyTransforms([0.0, 0.0, 0.3048], [0.0, 0.0, 0.0, 1.0], target_pos_on_arm, [0, 0, 0, 1], physicsClientId=self.id)[0])
            self.targets_pos_upperarm_world.append(target_pos)
            target.set_base_pos_orient(target_pos, [0, 0, 0, 1])

        if self.human.human_type == 'kinematic':
            forearm_pos, forearm_orient = self.human.get_pos_orient(self.forearm)
        elif self.human.human_type == 'mesh':
            forearm_pos, forearm_orient = self.human.get_mesh_pos_orient(self.forearm)

        self.targets_pos_forearm_world = []
        for target_pos_on_arm, target in zip(self.targets_pos_on_forearm, self.targets_forearm):
            target_pos = np.array(p.multiplyTransforms(forearm_pos, forearm_orient, target_pos_on_arm, [0, 0, 0, 1], physicsClientId=self.id)[0])
            self.targets_pos_forearm_world.append(target_pos)
            target.set_base_pos_orient(target_pos, [0, 0, 0, 1])

