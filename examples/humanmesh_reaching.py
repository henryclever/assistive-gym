import pybullet as p
import numpy as np
from assistive_gym.envs.env import AssistiveEnv
from assistive_gym.envs.agents.pr2 import PR2
from assistive_gym.envs.agents.agent import Agent
import assistive_gym.envs.agents.human_mesh as h
import time as time
from lib_update_mesh import LibUpdateMesh
from lib_pose_est.pose_estimator import PoseEstimator
from lib_pose_est.kinematics_lib_ag import KinematicsLib





class HumanMeshReaching():
    def __init__(self):
        dataset_info_dict = {}
        dataset_info_dict['gender'] = 'female'
        dataset_info_dict['data_ct'] = [60, 2000, 2112]  # 57 OK
        dataset_info_dict['posture'] = 'general_supine'
        dataset_info_dict['set_num'] = 10
        dataset_info_dict['mesh_type'] = 'ground_truth'
        self.human = h.HumanMesh(dataset_info_dict)

        dataset_info_dict_est = {}
        dataset_info_dict_est['gender'] = 'female'
        dataset_info_dict_est['data_ct'] = [60, 2000, 2112]
        dataset_info_dict_est['posture'] = 'general_supine'
        dataset_info_dict_est['set_num'] = 10
        dataset_info_dict_est['mesh_type'] = 'estimate'
        self.human_est = h.HumanMesh(dataset_info_dict_est)

        PE = PoseEstimator(dataset_info_dict)
        m, smpl_verts, joint_locs_trans_abs = PE.estimate_pose()

        self.human.assign_new_pose(m, smpl_verts, joint_locs_trans_abs)
        self.human_est.assign_new_pose(m, smpl_verts, joint_locs_trans_abs)

        robot_arm = 'left'
        self.robot = PR2(robot_arm, human_type='mesh')

        if True:
            LUM = LibUpdateMesh(dataset_info_dict)
            LUM.update_mattress_mesh()
            LUM.update_gt_human_mesh()
            # LUM.update_est_human_mesh(m, root_shift=root_shift)
            PE.update_est_human_mesh()

        self.env = AssistiveEnv(robot=self.robot, human=self.human, human_est=self.human_est, task='bed_bathing', bed_type='pressuresim',
                           render=True)
        self.env.reset()

        print('building assistive env')
        self.robot, self.human, self.human_est, furniture = self.env.build_assistive_env(furniture_type='bed_mesh', fixed_human_base=False)

        furniture.set_friction(furniture.base, friction=5)





        human_joint_goal_loc = self.human.get_mesh_pos_orient(self.human.right_hip)[0] + np.array([0.0, 0.0, 0.06])
        #human_joint_goal_loc[1] = 0.87

        human_joint_goal_loc = self.human.get_random_mesh_loc() + np.array([0.0, 0.0, 0.05])
        #4984 is on the right hip and has a error where the angle is too tilted so the robots arm gets hung up  on the bed.


        np_random = self.env.np_random


        if robot_arm == 'left':
            target_ee_pos = np.array([-0.5, human_joint_goal_loc[1]+0.3, 0.9]) #+ np_random.uniform(-0.05, 0.05, size=3)
            #target_ee_pos = np.array([-0.5, 0.2, 0.9]) #+ np_random.uniform(-0.05, 0.05, size=3)
        if robot_arm == 'right':
            target_ee_pos = np.array([-0.5, human_joint_goal_loc[1]-0.3, 0.9]) #+ np_random.uniform(-0.05, 0.05, size=3)

        else:
            assert('robot arm name not valid')

        print('target ee orient', self.env.task, self.env.id)
        print(target_ee_pos, self.robot.toc_ee_orient_rpy)
        target_ee_orient = np.array(
            p.getQuaternionFromEuler(np.array(self.robot.toc_ee_orient_rpy[self.env.task]), physicsClientId=self.env.id))

        # Use TOC with JLWKI to find an optimal base position for the robot near the person
        base_position, _, _ = self.robot.position_robot_toc(self.env.task, robot_arm, [(target_ee_pos, target_ee_orient)],
                                                       #[(shoulder_pos, None), (elbow_pos, None), (wrist_pos, None)],
                                                       [(human_joint_goal_loc, None)],
                                                       self.human, step_sim=True, check_env_collisions=False, right_side = True)

        print(base_position, 'base position')

        # Open gripper to hold the tool
        self.robot.set_gripper_open_position(self.robot.left_gripper_indices, self.robot.gripper_pos[self.env.task], set_instantly=True)
        # Initialize the tool in the robot's gripper
        self.env.tool.init(self.robot, self.env.task, self.env.directory, self.env.id, np_random, right=False, mesh_scale=[1] * 3)

        self.generate_targets()

        p.setGravity(0, 0, -9.81, physicsClientId=self.env.id)
        self.robot.set_gravity(0, 0, 0)
        # human.set_gravity(0, 0, -1)
        self.env.tool.set_gravity(0, 0, 0)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.env.id)

        self.env.init_env_variables()
        self.time_last = time.time()

        self.env.tool.enable_force_torque_sensor(0)

        pos, orient = self.robot.get_pos_orient(self.robot.left_end_effector)

        self.pose_seq = [(human_joint_goal_loc, orient)]
        self.pose_seq = self.get_pretouch_tool_path(self.env.tool.get_pos_orient(1), self.pose_seq[0])



        self.goal_pos_obj_list = self.env.create_surface_normal(pos=self.pose_seq[0][0], orient=self.pose_seq[0][1],
                                                      rgba=[0.7, 0.7, 0.15, 0.5])
        #end_eff_rev_offset = np.matmul(KinematicsLib().quaternionToRotationMatrix(self.pose_seq[0][1]), np.array([0.0, 0.0, 0.05]))
        #self.goal_pos_obj_list2 = self.env.create_surface_normal(pos=self.pose_seq[0][0] + end_eff_rev_offset,
        #                                               orient=self.pose_seq[0][1], rgba=[0.7, 0.0, 0.15, 0.5])

        self.human_surf_goal_obj_list = self.env.create_surface_normal(pos=self.pose_seq[-1][0], orient=self.pose_seq[-1][1],
                                                             rgba=[0.1, 0.9, 0.15, 0.5])

        self.target_joint_angles = self.robot.ik(self.robot.left_end_effector, self.pose_seq[0][0], self.pose_seq[0][1],
                                       ik_indices=self.robot.left_arm_ik_indices, max_iterations=1000)
        self.tool_force = 0
        self.contact_normal = None
        self.has_updated_init_touch = False
        self.init_touch_ct = 0
        self.offset_dist = 0.03

        # To initially move tool on a straight path, create a series of target steps.
        print(self.env.tool.get_pos_orient(1)[0], self.pose_seq[0][0],
              np.linalg.norm(self.env.tool.get_pos_orient(1)[0] - self.pose_seq[0][0]), 'total path length')



    def run_sim_iteration(self):

        robot_joint_angles = self.robot.get_joint_angles(self.robot.left_arm_joint_indices)

        # Before contact is made, ensure the tool moves along a relatively straight path.
        if self.tool_force == 0 and self.has_updated_init_touch == False and np.linalg.norm(
                self.target_joint_angles - robot_joint_angles) < 0.03 and len(self.pose_seq) > 1:
            del (self.pose_seq[0])
            print("updating pre-touch tool path")

            updated_goal_pos = self.pose_seq[0][0]
            updated_goal_orient = self.pose_seq[0][1]

            end_eff_rev_offset = np.array([0.0, 0.0, self.offset_dist])

            for item in self.goal_pos_obj_list:
                item.set_base_pos_orient(pos=updated_goal_pos, orient=updated_goal_orient)
            #for item in self.goal_pos_obj_list2:
            #    item.set_base_pos_orient(pos=updated_goal_pos + end_eff_rev_offset, orient=updated_goal_orient)
            self.target_joint_angles = self.robot.ik(self.robot.left_end_effector, updated_goal_pos + end_eff_rev_offset, updated_goal_orient,
                                           ik_indices=self.robot.left_arm_ik_indices, max_iterations=1000,
                                           use_current_as_rest=True)

        # Once contact has been made, set final position to current position and rotate end effector to be parallel with skin
        # we need to do this a few times though to balance it out.
        elif self.tool_force > 0 and self.has_updated_init_touch == False:  # np.linalg.norm(target_joint_angles - robot_joint_angles) < 0.01 and len(self.pose_seq) > 1:

            # Check if joint angles are close to the target, if so, select next target
            if self.init_touch_ct < 3:
                self.init_touch_ct += 1
            else:

                end_eff_rev_offset = self.offset_dist * np.array(self.contact_normal)

                for item in self.goal_pos_obj_list:
                    item.set_base_pos_orient(pos=self.contact_pos, orient=self.updated_goal_orient)
                #for item in self.goal_pos_obj_list2:
                #    item.set_base_pos_orient(pos=self.contact_pos + end_eff_rev_offset, orient=self.updated_goal_orient)
                print("updating end effector orientation")
                self.target_joint_angles = self.robot.ik(self.robot.left_end_effector, self.contact_pos + end_eff_rev_offset, self.updated_goal_orient,
                                               ik_indices=self.robot.left_arm_ik_indices, max_iterations=1000,
                                               use_current_as_rest=True)
                # target_joint_angles = self.robot.ik(self.robot.left_end_effector, tool_pos, self.updated_goal_orient, ik_indices=self.robot.left_arm_ik_indices, max_iterations=1000, use_current_as_rest=True)
                self.has_updated_init_touch = True



                # Follow a circle around the original point
                radius = 0.03
                self.pose_seq = []

                for theta in np.linspace(0, 4 * np.pi, 100):
                    self.pose_seq = self.append_from_index(self.pose_seq, self.contact_pos, self.updated_euler, np.array([radius * np.cos(theta), radius * np.sin(theta), 0]))


        elif self.has_updated_init_touch == True:

            if self.init_touch_ct < 50:
                self.init_touch_ct += 1
            else:
                #if np.linalg.norm(self.target_joint_angles - robot_joint_angles) < 0.03 and len(self.pose_seq) > 1:
                #print(self.tool_pos, self.pose_seq[0][0], np.linalg.norm(self.tool_pos - self.pose_seq[0][0]))
                if np.linalg.norm(self.tool_pos - self.pose_seq[0][0]) < 0.05 and len(self.pose_seq) > 1:
                    updated_goal_pos = self.pose_seq[0][0]
                    updated_goal_orient = self.pose_seq[0][1]

                    end_eff_rev_offset = np.array([0.0, 0.0, self.offset_dist])

                    for item in self.goal_pos_obj_list:
                        item.set_base_pos_orient(pos=updated_goal_pos, orient=updated_goal_orient)
                    # for item in self.goal_pos_obj_list2:
                    #    item.set_base_pos_orient(pos=updated_goal_pos + end_eff_rev_offset, orient=updated_goal_orient)
                    self.target_joint_angles = self.robot.ik(self.robot.left_end_effector,
                                                             updated_goal_pos + end_eff_rev_offset, updated_goal_orient,
                                                             ik_indices=self.robot.left_arm_ik_indices, max_iterations=1000,
                                                             use_current_as_rest=True)

                    del (self.pose_seq[0])





        joint_action = (self.target_joint_angles - robot_joint_angles) * 20
        # print('gains:', env.config('robot_gains'))
        self.env.take_step(joint_action, gains=self.env.config('robot_gains'), forces=self.env.config('robot_forces'))
        # self.env.take_step(joint_action, gains=0.2, forces=self.env.config('robot_forces'))
        pos, orient = self.robot.get_pos_orient(self.robot.left_end_effector)

        force_direction = np.array(self.env.tool.get_force_torque_sensor(0)[0:3]) / np.linalg.norm(
            self.env.tool.get_force_torque_sensor(0)[0:3])

        if self.contact_normal is not None:
            #self.updated_euler, self.updated_goal_orient, self.updated_R = self.get_updated_euler_quat(force_direction, orient)
            self.updated_euler, self.updated_goal_orient, self.updated_R = self.get_updated_euler_quat(-np.array(self.contact_normal), orient)
            print("force dir: ", -force_direction, "     surface normal:", self.contact_normal)

        print('End effector position error: %.2fcm' % (np.linalg.norm(pos - self.pose_seq[0][0]) * 100),
              'Orientation error:', ['%.4f' % v for v in (orient - self.pose_seq[0][1])])
        #print(np.linalg.norm(self.target_joint_angles - robot_joint_angles))

        self.tool_pos, tool_orient = self.env.tool.get_pos_orient(1)

        tool_pos_real, tool_orient_real = self.robot.convert_to_realworld(self.tool_pos, tool_orient)
        robot_joint_angles = self.robot.get_joint_angles(self.robot.controllable_joint_indices)

        self.tool_force, tool_force_on_human, total_force_on_human, self.contact_pos, self.contact_normal, new_contact_points = self.get_total_force()
        if self.contact_normal is not None:
            #self.contact_pos += np.array([0,0,0.1])
            #self.contact_normal = np.array([0, -0.001, 1])
            self.contact_normal = self.contact_normal/np.linalg.norm(self.contact_normal)

        print("Tool pos.:", self.tool_pos, "   Tool force:", self.tool_force, "   Contact normal:", self.contact_normal,
              "   Contact points:", new_contact_points)




    def generate_targets(self):
        self.target_indices_to_ignore = []

        self.target_pos_global_joints = []
        for i in range(24): #right now this is hacked to show markers at every joint rather than around the upper arm
            pos, _ = self.human.get_mesh_pos_orient(i)
            #print(pos)
            self.target_pos_global_joints.append(np.array(pos))


        self.targets_joints = self.env.create_spheres(radius=0.05, mass=0.0, batch_positions=[[0, 0, 0]]*len(self.target_pos_global_joints), visual=True, collision=False, rgba=[1, 0.2, 0.2, 1])

        self.total_target_count = len(self.target_pos_global_joints)
        self.update_targets()

    def update_targets(self):

        self.targets_pos_alljoints_world = []
        for target_pos_global, target in zip(self.target_pos_global_joints, self.targets_joints):
            target_pos = np.array(p.multiplyTransforms([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], target_pos_global, [0, 0, 0, 1], physicsClientId=self.env.id)[0])
            self.targets_pos_alljoints_world.append(target_pos)
            target.set_base_pos_orient(target_pos, [0, 0, 0, 1])


    def get_total_force(self):
        total_force_on_human = np.sum(self.robot.get_contact_points(self.human)[-1])
        tool_force = np.sum(self.env.tool.get_contact_points()[-1])
        contact_normal_list = self.env.tool.get_contact_points()[-2]
        contact_pos_list = self.env.tool.get_contact_points()[-4]
        if len(contact_normal_list) > 0:
            contact_pos = list(contact_pos_list[0])
            contact_normal = list(contact_normal_list[0])
        else:
            contact_pos = None
            contact_normal = None

        #print("tool contact points: ", self.env.tool.get_contact_points())
        #print("tool contact points: ", self.env.tool.get_contact_points(human))

        tool_force_on_human = 0
        new_contact_points = 0
        for linkA, linkB, posA, posB, normal, force in zip(*self.env.tool.get_contact_points(self.human)):
            total_force_on_human += force
            if linkA in [1]:
                tool_force_on_human += force
                # Only consider contact with human upperarm, forearm, hand
                if linkB < 0 or linkB > len(self.human.all_joint_indices):
                    continue


                indices_to_delete = []
                for i, (target_pos_world, target) in enumerate(zip(self.targets_pos_alljoints_world, self.targets_joints)):
                    if np.linalg.norm(posB - target_pos_world) < 0.025:
                        # The robot made contact with a point on the person's arm
                        new_contact_points += 1
                        self.task_success += 1
                        target.set_base_pos_orient(self.np_random.uniform(1000, 2000, size=3), [0, 0, 0, 1])
                        indices_to_delete.append(i)
                self.target_pos_global_joints = [t for i, t in enumerate(self.target_pos_global_joints) if i not in indices_to_delete]
                self.targets_joints = [t for i, t in enumerate(self.targets_joints) if i not in indices_to_delete]
                self.targets_pos_alljoints_world = [t for i, t in enumerate(self.targets_pos_alljoints_world) if i not in indices_to_delete]


        #print(tool_force, tool_force_on_human, total_force_on_human)
        return tool_force, tool_force_on_human, total_force_on_human, contact_pos, contact_normal, new_contact_points



    def get_updated_euler_quat(self, force_direction, orient, verbose = False):


        current_euler_angles = KinematicsLib().quaternionToEulerAngles(orient)

        diff_vector = -force_direction #- np.array([0.0, 0.0, -1.0])
        diff_vector = diff_vector/np.linalg.norm(diff_vector)
        updated_euler_angles = KinematicsLib().vectorToEulerAngles(diff_vector)

        if verbose == True:
            print(force_direction, current_euler_angles, updated_euler_angles, 'force dir, current and new eulers')

        updated_orient = KinematicsLib().eulerAnglesToQuaternion(updated_euler_angles)


        updated_R = KinematicsLib().eulerAnglesToRotationMatrix([updated_euler_angles[2], updated_euler_angles[0], updated_euler_angles[1]] ) #WORLD!! not tool frame.

        return updated_euler_angles, updated_orient, updated_R


    def append_from_index(self, pose_sequence, pos_goal, euler_angles_goal, pos_add):
        R = KinematicsLib().eulerAnglesToRotationMatrix(euler_angles_goal)
        orient = KinematicsLib().eulerAnglesToQuaternion(euler_angles_goal)

        rotated_point = np.matmul(R, np.array([[pos_add[2]], [pos_add[1]], [pos_add[0]]])).T[0]
        rotated_point = np.array([rotated_point[2], -rotated_point[1], -rotated_point[0]])

        pose_sequence.append((pos_goal + rotated_point, orient))
        return pose_sequence



    def get_pretouch_tool_path(self, curr_pose_seq, goal_pose_seq):
        num_points_to_add = int(np.linalg.norm(curr_pose_seq[0] - goal_pose_seq[0])*10)*2

        modded_pose_seq = []
        for point_to_add in range(num_points_to_add):
            next_position_to_add = curr_pose_seq[0] + (point_to_add+1)*(goal_pose_seq[0] - curr_pose_seq[0])/(1+num_points_to_add)
            modded_pose_seq.append((next_position_to_add , goal_pose_seq[1]))

        modded_pose_seq.append(goal_pose_seq)
        return modded_pose_seq



if __name__ == '__main__':

    HMR = HumanMeshReaching()


    for step_num in range(1000):
        print('taking step', step_num, '  time: ', time.time()-HMR.time_last)
        time_last = time.time()
        HMR.run_sim_iteration()



    p.disconnect(self.env.id)

