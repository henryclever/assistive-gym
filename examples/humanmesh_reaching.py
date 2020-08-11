import pybullet as p
import numpy as np
from assistive_gym.envs.env import AssistiveEnv
from assistive_gym.envs.agents.pr2 import PR2
from assistive_gym.envs.agents.agent import Agent
import assistive_gym.envs.agents.human_mesh as h
import time as time
from lib_update_mesh import LibUpdateMesh
from lib_pose_est.pose_estimator import PoseEstimator

#from gibson2.core.physics.scene import BuildingScene


class HumanMeshReaching():
    def __init__(self):
        pass

    def generate_targets(self):
        self.target_indices_to_ignore = []
        if human.gender == 'male':
            self.upperarm, self.upperarm_length, self.upperarm_radius = human.right_shoulder, 0.279, 0.04
        else:
            self.upperarm, self.upperarm_length, self.upperarm_radius = human.right_shoulder, 0.264, 0.0355

        self.targets_pos_on_upperarm = env.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -self.upperarm_length]), radius=self.upperarm_radius, distance_between_points=0.03)

        self.target_pos_global_joints = []
        for i in range(24): #right now this is hacked to show markers at every joint rather than around the upper arm
            pos, _ = human.get_mesh_pos_orient(i)
            print(pos)
            self.target_pos_global_joints.append(np.array(pos))


        #for idx in range(np.shape(self.targets_pos_on_upperarm)[0]):
        #    self.targets_pos_on_upperarm[idx][0] = 0.0
        #    self.targets_pos_on_upperarm[idx][1] = 0.0
        #    self.targets_pos_on_upperarm[idx][2] = 0.0



        self.targets_upperarm = env.create_spheres(radius=0.05, mass=0.0, batch_positions=[[0, 0, 0]]*len(self.targets_pos_on_upperarm), visual=True, collision=False, rgba=[0.2, 0.6, 0.2, 1])
        self.targets_joints = env.create_spheres(radius=0.05, mass=0.0, batch_positions=[[0, 0, 0]]*len(self.target_pos_global_joints), visual=True, collision=False, rgba=[1, 0.2, 0.2, 1])


        self.total_target_count = len(self.targets_pos_on_upperarm) + len(self.target_pos_global_joints)
        self.update_targets()

    def update_targets(self):


        #upperarm_pos, upperarm_orient = self.human.get_mesh_pos_orient(self.upperarm) ##### relative to upper arm! need to get this right.
        upperarm_pos, upperarm_orient = human.get_mesh_pos_orient(0) ##### relative to upper arm! need to get this right.
        #print(upperarm_pos)

        self.targets_pos_upperarm_world = []
        for target_pos_on_arm, target in zip(self.targets_pos_on_upperarm, self.targets_upperarm):
            #target_pos = np.array(p.multiplyTransforms(upperarm_pos, upperarm_orient, target_pos_on_arm, [0, 0, 0, 1], physicsClientId=self.id)[0])
            target_pos = np.array(p.multiplyTransforms([0.0, 0.0, 0.3048], [0.0, 0.0, 0.0, 1.0], target_pos_on_arm, [0, 0, 0, 1], physicsClientId=env.id)[0])
            self.targets_pos_upperarm_world.append(target_pos)
            target.set_base_pos_orient(target_pos, [0, 0, 0, 1])

        self.targets_pos_alljoints_world = []
        for target_pos_global, target in zip(self.target_pos_global_joints, self.targets_joints):
            target_pos = np.array(p.multiplyTransforms([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], target_pos_global, [0, 0, 0, 1], physicsClientId=env.id)[0])
            self.targets_pos_alljoints_world.append(target_pos)
            target.set_base_pos_orient(target_pos, [0, 0, 0, 1])


    def get_total_force(self):
        total_force_on_human = np.sum(robot.get_contact_points(human)[-1])
        tool_force = np.sum(env.tool.get_contact_points()[-1])
        tool_force_on_human = 0
        new_contact_points = 0
        for linkA, linkB, posA, posB, force in zip(*env.tool.get_contact_points(human)):
            total_force_on_human += force
            if linkA in [1]:
                tool_force_on_human += force
                # Only consider contact with human upperarm, forearm, hand
                if linkB < 0 or linkB > len(human.all_joint_indices):
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


        return tool_force, tool_force_on_human, total_force_on_human, new_contact_points




if __name__ == '__main__':

    dataset_info_dict = {}
    dataset_info_dict['gender'] = 'female'
    dataset_info_dict['data_ct'] = [60, 2000, 2112] #57 OK
    dataset_info_dict['posture'] = 'general_supine'
    dataset_info_dict['set_num'] = 10
    dataset_info_dict['mesh_type'] = 'ground_truth'
    human = h.HumanMesh(dataset_info_dict)

    dataset_info_dict_est = {}
    dataset_info_dict_est['gender'] = 'female'
    dataset_info_dict_est['data_ct'] = [60, 2000, 2112]
    dataset_info_dict_est['posture'] = 'general_supine'
    dataset_info_dict_est['set_num'] = 10
    dataset_info_dict_est['mesh_type'] = 'estimate'
    human_est = h.HumanMesh(dataset_info_dict_est)


    PE = PoseEstimator(dataset_info_dict)
    m, joint_locs_trans_abs = PE.estimate_pose()



    #human.load_smpl_model(joint_locs_trans_abs)
    #human_est.load_smpl_model(joint_locs_trans_abs)

    human.assign_new_pose(m, joint_locs_trans_abs)
    human_est.assign_new_pose(m, joint_locs_trans_abs)

    robot_arm = 'left'
    robot=PR2(robot_arm, human_type='mesh')
    #[0.14420165 0.48038295 0.09955744]

    HMR = HumanMeshReaching()


    if True:
        LUM = LibUpdateMesh(dataset_info_dict)
        LUM.update_mattress_mesh()
        LUM.update_gt_human_mesh()
        #LUM.update_est_human_mesh(m, root_shift=root_shift)
        PE.update_est_human_mesh()


    env = AssistiveEnv(robot = robot, human = human, human_est=human_est, task='bed_bathing', bed_type='pressuresim', render=True)
    env.reset()

    print('building assistive env')
    robot, human, human_est, furniture = env.build_assistive_env(furniture_type='bed_mesh', fixed_human_base=False)


    furniture.set_friction(furniture.base, friction=5)

    shoulder_pos = human.get_mesh_pos_orient(human.right_shoulder)[0]
    elbow_pos = human.get_mesh_pos_orient(human.right_elbow)[0]
    wrist_pos = human.get_mesh_pos_orient(human.right_wrist)[0]

    np_random = env.np_random

    target_ee_pos = np.array([-0.6, 0.2, 1]) + np_random.uniform(-0.05, 0.05, size=3)

    print('target ee orient', env.task, env.id)
    print(robot.toc_ee_orient_rpy)
    target_ee_orient = np.array(p.getQuaternionFromEuler(np.array(robot.toc_ee_orient_rpy[env.task]), physicsClientId=env.id))

    # Use TOC with JLWKI to find an optimal base position for the robot near the person
    base_position, _, _ = robot.position_robot_toc(env.task, 'left', [(target_ee_pos, target_ee_orient)],
                                                        [(shoulder_pos, None), (elbow_pos, None), (wrist_pos, None)],
                                                        human, step_sim=True, check_env_collisions=False)

    # Open gripper to hold the tool
    robot.set_gripper_open_position(robot.left_gripper_indices, robot.gripper_pos[env.task], set_instantly=True)
    # Initialize the tool in the robot's gripper
    env.tool.init(robot, env.task, env.directory, env.id, np_random, right=False, mesh_scale=[1] * 3)

    HMR.generate_targets()

    p.setGravity(0, 0, -9.81, physicsClientId=env.id)
    robot.set_gravity(0, 0, 0)
    human.set_gravity(0, 0, -1)
    env.tool.set_gravity(0, 0, 0)

    # Enable rendering
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=env.id)

    env.init_env_variables()

    time_last = time.time()

    def quat_rpy_add(quat, rpy):
        new_rpy = p.getEulerFromQuaternion(quat, physicsClientId=env.id) + rpy
        return p.getQuaternionFromEuler(new_rpy, physicsClientId=env.id)

    pos, orient = robot.get_pos_orient(robot.left_end_effector)

    human_joint_loc = human.get_mesh_pos_orient(5)[0]
    #human_joint_loc = pos # - np.array([0.2, -0.2, -0.2])
    #r_knee_pos =


    print('knee is close to', human_joint_loc)


    pose_seq = [(human_joint_loc, orient)]
    def append_from_index(pos_add, rpy_add=np.array([0, 0, 0]), index=-1):
        pose_seq.append((pose_seq[index][0] + pos_add, quat_rpy_add(pose_seq[index][1], rpy_add)))

    # Follow a circle around the original point
    radius = 0.1
    for theta in np.linspace(0, 4*np.pi, 100):
        append_from_index(np.array([radius*np.cos(theta), radius*np.sin(theta), 0]), index=0)

    # Reach four points with varying orientations
    # append_from_index(np.array([0, 0.1, 0]), np.array([-np.pi/4.0, 0, 0]))
    # append_from_index(np.array([0, -0.2, 0]), np.array([np.pi/2.0, 0, 0]))
    # append_from_index(np.array([0.1, 0.1, 0]), np.array([-np.pi/4.0, np.pi/4.0, 0]))
    # append_from_index(np.array([-0.2, 0, 0]), np.array([0, -np.pi/2.0, 0]))

    target_joint_angles = robot.ik(robot.left_end_effector, pose_seq[0][0], pose_seq[0][1], ik_indices=robot.left_arm_ik_indices, max_iterations=1000)

    for step_num in range(500):
        print('taking step', step_num, '  time: ', time.time()-time_last)
        time_last = time.time()
        robot_joint_angles = robot.get_joint_angles(robot.left_arm_joint_indices)
        if np.linalg.norm(target_joint_angles - robot_joint_angles) < 0.01 and len(pose_seq) > 1:
            # Check if joint angles are close to the target, if so, select next target
            del pose_seq[0]
            target_joint_angles = robot.ik(robot.left_end_effector, pose_seq[0][0], pose_seq[0][1], ik_indices=robot.left_arm_ik_indices, max_iterations=1000, use_current_as_rest=True)
        joint_action = (target_joint_angles - robot_joint_angles) * 10
        #print('gains:', env.config('robot_gains'))
        env.take_step(joint_action, gains=env.config('robot_gains'), forces=env.config('robot_forces'))
        #env.take_step(joint_action, gains=0.2, forces=env.config('robot_forces'))
        pos, orient = robot.get_pos_orient(robot.left_end_effector)

        #print(pos, pose_seq[0][0])

        print('End effector position error: %.2fcm' % (np.linalg.norm(pos - pose_seq[0][0])*100), 'Orientation error:', ['%.4f' % v for v in (orient - pose_seq[0][1])])
        # print(np.linalg.norm(target_joint_angles - robot_joint_angles))

        tool_pos, tool_orient = env.tool.get_pos_orient(1)
        tool_pos_real, tool_orient_real = robot.convert_to_realworld(tool_pos, tool_orient)
        robot_joint_angles = robot.get_joint_angles(robot.controllable_joint_indices)

        shoulder_pos = human.get_mesh_pos_orient(human.right_shoulder)[0]
        elbow_pos = human.get_mesh_pos_orient(human.right_elbow)[0]
        wrist_pos = human.get_mesh_pos_orient(human.right_wrist)[0]

        shoulder_pos_real, _ = robot.convert_to_realworld(shoulder_pos)
        elbow_pos_real, _ = robot.convert_to_realworld(elbow_pos)
        wrist_pos_real, _ = robot.convert_to_realworld(wrist_pos)
        tool_force, tool_force_on_human, total_force_on_human, new_contact_points = HMR.get_total_force()

        print(tool_force)

        robot_obs = np.concatenate(
            [tool_pos_real, tool_orient_real, robot_joint_angles, shoulder_pos_real, elbow_pos_real, wrist_pos_real,
             [tool_force]]).ravel()
        if human.controllable:
            human_obs = np.concatenate(
                [tool_pos_human, tool_orient_human, human_joint_angles, shoulder_pos_human, elbow_pos_human,
                 wrist_pos_human, [total_force_on_human, tool_force_on_human]]).ravel()
        else:
            human_obs = []

        obs = np.concatenate([robot_obs, human_obs]).ravel()




    p.disconnect(env.id)

