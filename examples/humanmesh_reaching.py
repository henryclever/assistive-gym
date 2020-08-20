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
        m, joint_locs_trans_abs = PE.estimate_pose()

        self.human.assign_new_pose(m, joint_locs_trans_abs)
        self.human_est.assign_new_pose(m, joint_locs_trans_abs)

        robot_arm = 'left'
        robot = PR2(robot_arm, human_type='mesh')

        if True:
            LUM = LibUpdateMesh(dataset_info_dict)
            LUM.update_mattress_mesh()
            LUM.update_gt_human_mesh()
            # LUM.update_est_human_mesh(m, root_shift=root_shift)
            PE.update_est_human_mesh()

        self.env = AssistiveEnv(robot=robot, human=self.human, human_est=self.human_est, task='bed_bathing', bed_type='pressuresim',
                           render=True)
        self.env.reset()

        print('building assistive env')
        robot, self.human, self.human_est, furniture = self.env.build_assistive_env(furniture_type='bed_mesh', fixed_human_base=False)

        furniture.set_friction(furniture.base, friction=5)

        shoulder_pos = self.human.get_mesh_pos_orient(self.human.right_shoulder)[0]
        elbow_pos = self.human.get_mesh_pos_orient(self.human.right_elbow)[0]
        wrist_pos = self.human.get_mesh_pos_orient(self.human.right_wrist)[0]

        np_random = self.env.np_random
        updated_R = np.identity(4)

        target_ee_pos = np.array([-0.5, 0.2, 1]) + np_random.uniform(-0.05, 0.05, size=3)

        print('target ee orient', self.env.task, self.env.id)
        print(target_ee_pos, robot.toc_ee_orient_rpy)
        target_ee_orient = np.array(
            p.getQuaternionFromEuler(np.array(robot.toc_ee_orient_rpy[self.env.task]), physicsClientId=self.env.id))

        # Use TOC with JLWKI to find an optimal base position for the robot near the person
        base_position, _, _ = robot.position_robot_toc(self.env.task, 'left', [(target_ee_pos, target_ee_orient)],
                                                       [(shoulder_pos, None), (elbow_pos, None), (wrist_pos, None)],
                                                       self.human, step_sim=True, check_env_collisions=False)

        # Open gripper to hold the tool
        robot.set_gripper_open_position(robot.left_gripper_indices, robot.gripper_pos[self.env.task], set_instantly=True)
        # Initialize the tool in the robot's gripper
        self.env.tool.init(robot, self.env.task, self.env.directory, self.env.id, np_random, right=False, mesh_scale=[1] * 3)

        self.generate_targets()

        p.setGravity(0, 0, -9.81, physicsClientId=self.env.id)
        robot.set_gravity(0, 0, 0)
        # human.set_gravity(0, 0, -1)
        self.env.tool.set_gravity(0, 0, 0)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.env.id)

        self.env.init_env_variables()
        self.time_last = time.time()

        self.env.tool.enable_force_torque_sensor(0)

        pos, orient = robot.get_pos_orient(robot.left_end_effector)
        updated_goal_orient = np.array([0.0, 0.0, 0.0, 1.0])

        human_joint_loc = self.human.get_mesh_pos_orient(5)[0] + np.array([0.0, 0.0, 0.06])
        # human_joint_loc = pos - np.array([0.1, -0.1, -0.1])

        print('knee is close to', human_joint_loc)
        # orient = KinematicsLib().eulerAnglesToQuaternion([-0.01479592,  0.0,  -2.73740005])#[0.0, 0.0, -np.pi]) #keep the first angle (for 'yaw') equal to zero. solve for the other two

        pose_seq = [(human_joint_loc, orient)]
        pose_seq = self.get_pretouch_tool_path(self.env.tool.get_pos_orient(1), pose_seq[0])



        self.goal_pos_obj_list = self.env.create_surface_normal(pos=pose_seq[0][0], orient=pose_seq[0][1],
                                                      rgba=[0.7, 0.7, 0.15, 0.5])

        wiper_offset_goal_pos = np.matmul(KinematicsLib().quaternionToRotationMatrix(pose_seq[0][1]),
                                          np.array([0.0, 0.0, 0.05]))

        self.goal_pos_obj_list2 = self.env.create_surface_normal(pos=pose_seq[0][0] + wiper_offset_goal_pos,
                                                       orient=pose_seq[0][1], rgba=[0.7, 0.0, 0.15, 0.5])
        self.human_surf_goal_obj_list = self.env.create_surface_normal(pos=pose_seq[-1][0], orient=pose_seq[-1][1],
                                                             rgba=[0.1, 0.9, 0.15, 0.5])

        self.target_joint_angles = robot.ik(robot.left_end_effector, pose_seq[0][0], pose_seq[0][1],
                                       ik_indices=robot.left_arm_ik_indices, max_iterations=1000)
        self.tool_force = 0
        self.contact_normal = None
        self.has_updated_init_touch = False
        self.orientation_parallel = False
        self.init_touch_ct = 0

        # To initially move tool on a straight path, create a series of target steps.
        print(self.env.tool.get_pos_orient(1)[0], pose_seq[0][0],
              np.linalg.norm(self.env.tool.get_pos_orient(1)[0] - pose_seq[0][0]), 'total path length')



    def run_sim_iteration(self):

        robot_joint_angles = robot.get_joint_angles(robot.left_arm_joint_indices)

        # print(tool_force)

        # [ 0.73441682 -0.59247261 -0.33107121] [0.0272707  0.06162594 3.05391137] [ 0.0272707   0.46689816 -2.722813  ]
        # [ 0.59625563 -0.33059906 -0.73156236] [ 0.04764206  0.46648589 -2.72260244] [ 0.04764206  0.32615633 -2.9529378 ] 2nd

        # Before contact is made, ensure the tool moves along a relatively straight path.
        if tool_force == 0 and has_updated_init_touch == False and np.linalg.norm(
                target_joint_angles - robot_joint_angles) < 0.05 and len(pose_seq) > 1:
            del (pose_seq[0])
            print("updating pre-touch tool path")

            updated_goal_pos = pose_seq[0][0]
            updated_goal_orient = pose_seq[0][1]

            wiper_offset_goal_pos = np.array([0.0, 0.0, 0.05])

            for item in goal_pos_obj_list:
                item.set_base_pos_orient(pos=updated_goal_pos, orient=updated_goal_orient)
            for item in goal_pos_obj_list2:
                item.set_base_pos_orient(pos=updated_goal_pos + wiper_offset_goal_pos, orient=updated_goal_orient)

            target_joint_angles = robot.ik(robot.left_end_effector, updated_goal_pos, updated_goal_orient,
                                           ik_indices=robot.left_arm_ik_indices, max_iterations=1000,
                                           use_current_as_rest=True)

        # Once contact has been made, set final position to current position and rotate end effector to be parallel with skin
        # we need to do this a few times though to balance it out.
        elif tool_force > 0 and has_updated_init_touch == False:  # np.linalg.norm(target_joint_angles - robot_joint_angles) < 0.01 and len(pose_seq) > 1:

            # Check if joint angles are close to the target, if so, select next target
            if init_touch_ct < 3:
                init_touch_ct += 1
            else:

                wiper_offset_goal_pos = 0.05 * np.array(contact_normal)
                print("****************WIPER ", wiper_offset_goal_pos)

                for item in goal_pos_obj_list:
                    item.set_base_pos_orient(pos=contact_pos, orient=updated_goal_orient)
                for item in goal_pos_obj_list2:
                    item.set_base_pos_orient(pos=contact_pos + wiper_offset_goal_pos, orient=updated_goal_orient)

                print("updating end effector orientation")
                target_joint_angles = robot.ik(robot.left_end_effector, contact_pos, updated_goal_orient,
                                               ik_indices=robot.left_arm_ik_indices, max_iterations=1000,
                                               use_current_as_rest=True)
                # target_joint_angles = robot.ik(robot.left_end_effector, tool_pos, updated_goal_orient, ik_indices=robot.left_arm_ik_indices, max_iterations=1000, use_current_as_rest=True)
                has_updated_init_touch = True
                force_during_contact = np.copy(force_direction)
                updated_goal_orient_during_contact = np.copy(updated_goal_orient)
                tool_pos_during_init_contact = np.copy(tool_pos)


        # after updating orientation, check if there is any force. If there is, update the orientation again. If not, move closer.
        elif False:  # has_updated_init_touch == True and orientation_parallel == False:

            if init_touch_ct < 30:
                init_touch_ct += 1
            else:
                if tool_force == 0:
                    print("moving closer to skin")

                    tool_pos_during_init_contact = tool_pos_during_init_contact + np.array(force_during_contact) * 0.01

                    target_joint_angles = robot.ik(robot.left_end_effector, tool_pos_during_init_contact,
                                                   updated_goal_orient_during_contact,
                                                   ik_indices=robot.left_arm_ik_indices, max_iterations=1000,
                                                   use_current_as_rest=True)

                else:
                    print("updating end effector orientation again")
                    target_joint_angles = robot.ik(robot.left_end_effector, tool_pos_during_init_contact,
                                                   updated_goal_orient,
                                                   ik_indices=robot.left_arm_ik_indices, max_iterations=1000,
                                                   use_current_as_rest=True)

                    # create profile to Follow a circle around the original point
                    radius = 0.05
                    pose_seq = [
                        pose_seq[-1]]  # delete everything but the original final goal position from the sequence
                    for theta in np.linspace(0, 4 * np.pi, 100):
                        # append_from_index(pose_seq, np.array([radius * np.cos(theta), radius * np.sin(theta), 0]), index=0)
                        # pose_seq = HMR.append_from_index(pose_seq, [(tool_pos, updated_goal_orient)],
                        #                                 np.array([radius * np.cos(theta), radius * np.sin(theta), 0]), index=0)
                        pose_seq = self.append_from_index(pose_seq, [(contact_pos, updated_goal_orient)],
                                                         np.array([radius * np.cos(theta), radius * np.sin(theta), 0]),
                                                         index=0)
                    orientation_parallel = True


        # once orientation is OK, wait briefly for tool to stabilize
        elif False:  # has_updated_init_touch == True and init_touch_ct < 50 and orientation_parallel == True:
            init_touch_ct += 1


        # After contact has been made a short period of time, move in a circular profile
        # elif has_updated_init_touch == True and init_touch_ct >= 20 and np.linalg.norm(target_joint_angles - robot_joint_angles) < 0.01 and len(pose_seq) > 1:
        elif has_updated_init_touch == True and init_touch_ct >= 50 and orientation_parallel == True and np.linalg.norm(
                target_joint_angles - robot_joint_angles) < 0.1 and len(pose_seq) > 1:

            # if init_touch_ct == 20:
            del pose_seq[0]

            updated_goal_pos = pose_seq[0][0]
            updated_goal_orient = pose_seq[0][1]

            target_joint_angles = robot.ik(robot.left_end_effector, updated_goal_pos, updated_goal_orient,
                                           ik_indices=robot.left_arm_ik_indices, max_iterations=1000,
                                           use_current_as_rest=True)

            init_touch_ct += 1

        # print(np.linalg.norm(target_joint_angles - robot_joint_angles), 'norm target robot joint angles')

        joint_action = (target_joint_angles - robot_joint_angles) * 20
        # print('gains:', env.config('robot_gains'))
        self.env.take_step(joint_action, gains=self.env.config('robot_gains'), forces=self.env.config('robot_forces'))
        # self.env.take_step(joint_action, gains=0.2, forces=self.env.config('robot_forces'))
        pos, orient = robot.get_pos_orient(robot.left_end_effector)

        force_direction = np.array(self.env.tool.get_force_torque_sensor(0)[0:3]) / np.linalg.norm(
            self.env.tool.get_force_torque_sensor(0)[0:3])
        # print(force_direction)

        if contact_normal is not None:
            # updated_euler, updated_goal_orient, updated_R = self.get_updated_euler_quat(force_direction, orient)
            updated_euler, updated_goal_orient, updated_R = self.get_updated_euler_quat(-np.array(contact_normal),
                                                                                       orient)
            print("force dir: ", -force_direction, "     surface normal:", contact_normal)

        print('End effector position error: %.2fcm' % (np.linalg.norm(pos - pose_seq[0][0]) * 100),
              'Orientation error:', ['%.4f' % v for v in (orient - pose_seq[0][1])])
        print(np.linalg.norm(target_joint_angles - robot_joint_angles))

        tool_pos, tool_orient = self.env.tool.get_pos_orient(1)

        tool_pos_real, tool_orient_real = robot.convert_to_realworld(tool_pos, tool_orient)
        robot_joint_angles = robot.get_joint_angles(robot.controllable_joint_indices)

        tool_force, tool_force_on_human, total_force_on_human, contact_pos, contact_normal, new_contact_points = self.get_total_force()

        print("Tool pos.:", tool_pos, "   Tool force:", tool_force, "   Contact normal:", contact_normal,
              "   Contact points:", new_contact_points)




    def generate_targets(self):
        self.target_indices_to_ignore = []

        self.target_pos_global_joints = []
        for i in range(24): #right now this is hacked to show markers at every joint rather than around the upper arm
            pos, _ = self.human.get_mesh_pos_orient(i)
            print(pos)
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
        total_force_on_human = np.sum(robot.get_contact_points(self.human)[-1])
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

        #force_direction = np.array([0.00001, 0.00001, -1.0])
        diff_vector = -force_direction - np.array([0.0, 0.0, -1.0])

        diff_vector = diff_vector/np.linalg.norm(diff_vector)
        #print(diff_vector)
        if diff_vector[1] < 0:
            yaw = (np.pi + np.arctan(diff_vector[1]/diff_vector[2]))
            yaw2 = np.arctan(diff_vector[1]/diff_vector[2])
        else:
            yaw = -(np.pi - np.arctan(diff_vector[1]/diff_vector[2]))
            yaw2 = -np.arctan(diff_vector[1]/diff_vector[2])

        pitch = np.arccos(-diff_vector[2]/np.cos(yaw))

        current_euler_angles = KinematicsLib().quaternionToEulerAngles(orient)
        updated_euler_angles = np.array([current_euler_angles[0], pitch, yaw])

        if verbose == True:
            print(force_direction, current_euler_angles, updated_euler_angles, 'force dir, current and new eulers')

        updated_orient = KinematicsLib().eulerAnglesToQuaternion(updated_euler_angles)

        updated_R = KinematicsLib().eulerAnglesToRotationMatrix([yaw2, updated_euler_angles[0], updated_euler_angles[1]] ) #WORLD!! not tool frame.

        return updated_euler_angles, updated_orient, updated_R


    def quat_rpy_add(quat, rpy):
        new_rpy = p.getEulerFromQuaternion(quat, physicsClientId=self.env.id) + rpy
        return p.getQuaternionFromEuler(new_rpy, physicsClientId=self.env.id)


    def append_from_index(self, pose_sequence, pos_orient, pos_add, rpy_add=np.array([0, 0, 0]), index=-1):

        rotated_point = np.matmul(updated_R, np.array([[pos_add[0]], [pos_add[1]], [pos_add[2]]])).T[0]
        #rotated_point = np.matmul(np.array([pos_orient[index][0]]),updated_R)
        #print(pos_add, rotated_point, pos_add - rotated_point)

        #pose_seq.append((pos_orient[index][0] + pos_add, self.quat_rpy_add(pos_orient[index][1], rpy_add)))
        pose_sequence.append((pos_orient[index][0] + rotated_point, self.quat_rpy_add(pos_orient[index][1], rpy_add)))

        #print(pose_seq[-1][0], rotated_point+pos_add, 'updated goal pos')
        return pose_sequence

    def get_pretouch_tool_path(self, curr_pose_seq, goal_pose_seq):
        num_points_to_add = int(np.linalg.norm(curr_pose_seq[0] - goal_pose_seq[0])*10)*2

        modded_pose_seq = []
        for point_to_add in range(num_points_to_add):
            next_position_to_add = curr_pose_seq[0] + (point_to_add+1)*(goal_pose_seq[0] - curr_pose_seq[0])/(1+num_points_to_add)
            modded_pose_seq.append((next_position_to_add , goal_pose_seq[1]))

        modded_pose_seq.append(goal_pose_seq)

        #print(num_points_to_add)

        #for item in modded_pose_seq:
        #    print (item)
        return modded_pose_seq



if __name__ == '__main__':

    HMR = HumanMeshReaching()


    for step_num in range(1000):
        print('taking step', step_num, '  time: ', time.time()-HMR.time_last)
        time_last = time.time()
        HMR.run_sim_iteration()



    p.disconnect(self.env.id)

