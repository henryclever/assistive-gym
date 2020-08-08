import pybullet as p
import numpy as np
from assistive_gym.envs.env import AssistiveEnv
from assistive_gym.envs.agents.pr2 import PR2
from assistive_gym.envs.agents.agent import Agent
import assistive_gym.envs.agents.human as h
import time as time
#from gibson2.core.physics.scene import BuildingScene


class HumanMeshReaching():
    def __init__(self):
        pass

    def update_mattress_mesh(self, dataset_info_dict):
        print("****Updating Mattress and Pressure Mat Mesh****")
        mesh_data_folder = "/home/henry/data/resting_meshes/"+dataset_info_dict['posture']+\
                           "/roll0_"+dataset_info_dict['gender'][0]+\
                           "_lay_set"+str(dataset_info_dict['set_num'])+\
                           "_"+str(dataset_info_dict['data_ct'][1])+\
                           "_of_"+str(dataset_info_dict['data_ct'][2])+"_none_stiff/"

        mv = np.load(mesh_data_folder + "mv.npy")  # , allow_pickle = True, encoding='latin1')
        mf = np.load(mesh_data_folder + "mf.npy")  # , allow_pickle = True, encoding='latin1')
        bv = np.load(mesh_data_folder + "bv.npy", allow_pickle=True, encoding='latin1')
        bf = np.load(mesh_data_folder + "bf.npy", allow_pickle=True, encoding='latin1')

        for sample_idx in range( dataset_info_dict['data_ct'][0],  dataset_info_dict['data_ct'][0]+1):  # (0, np.shape(mv)[0]):
            mattress_verts = np.array(mv[sample_idx, :, :]) / 2.58872  # np.load(folder + "mattress_verts_py.npy")/2.58872
            mattress_verts = np.concatenate((mattress_verts[:, 2:3], mattress_verts[:, 0:1], mattress_verts[:, 1:2]),
                                            axis=1)

            mattress_faces = np.array(mf[sample_idx, :, :])  # np.load(folder + "mattress_faces_py.npy")
            mattress_faces = np.concatenate((np.array([[0, 6054, 6055]]), mattress_faces), axis=0)
            # mattress_vf = [mattress_verts, mattress_faces]

            mat_blanket_verts = np.array(bv[sample_idx]) / 2.58872  # np.load(folder + "mat_blanket_verts_py.npy")/2.58872
            mat_blanket_verts = np.concatenate(
                (mat_blanket_verts[:, 2:3], mat_blanket_verts[:, 0:1], mat_blanket_verts[:, 1:2]), axis=1)
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
                fp.write('f %d %d %d\n' % (
                mattress_faces[f_idx, 0] + 1, mattress_faces[f_idx, 1] + 1, mattress_faces[f_idx, 2] + 1))

        outmesh_pmat_path = "/home/henry/git/assistive-gym/assistive_gym/envs/assets/bed_mesh/bed_pmat.obj"
        with open(outmesh_pmat_path, 'w') as fp:
            for v_idx in range(pmat_verts.shape[0]):
                fp.write('v %f %f %f\n' % (pmat_verts[v_idx, 0], pmat_verts[v_idx, 1], pmat_verts[v_idx, 2]))

            for f_idx in range(pmat_faces.shape[0]):
                fp.write('f %d %d %d\n' % (pmat_faces[f_idx, 0] + 1, pmat_faces[f_idx, 1] + 1, pmat_faces[f_idx, 2] + 1))


    def generate_targets(self):
        self.target_indices_to_ignore = []
        if human.gender == 'male':
            self.upperarm, self.upperarm_length, self.upperarm_radius = human.right_shoulder, 0.279, 0.043
            self.forearm, self.forearm_length, self.forearm_radius = human.right_elbow, 0.257, 0.033
        else:
            self.upperarm, self.upperarm_length, self.upperarm_radius = human.right_shoulder, 0.264, 0.0355
            self.forearm, self.forearm_length, self.forearm_radius = human.right_elbow, 0.234, 0.027

        self.targets_pos_on_upperarm = env.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -self.upperarm_length]), radius=self.upperarm_radius, distance_between_points=0.03)
        self.targets_pos_on_forearm = env.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -self.forearm_length]), radius=self.forearm_radius, distance_between_points=0.03)

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
            pos, _ = human.get_mesh_pos_orient(i)
            self.targets_pos_on_upperarm[i][0] = pos[0]
            self.targets_pos_on_upperarm[i][1] = pos[1]
            self.targets_pos_on_upperarm[i][2] = pos[2]


        self.targets_upperarm = env.create_spheres(radius=0.05, mass=0.0, batch_positions=[[0, 0, 0]]*len(self.targets_pos_on_upperarm), visual=True, collision=False, rgba=[1, 0.2, 0.2, 1])
        self.targets_forearm = env.create_spheres(radius=0.01, mass=0.0, batch_positions=[[0, 0, 0]]*len(self.targets_pos_on_forearm), visual=True, collision=False, rgba=[0, 1, 1, 0])

        #print(self.targets_pos_on_forearm)
        self.total_target_count = len(self.targets_pos_on_upperarm) + len(self.targets_pos_on_forearm)
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

        forearm_pos, forearm_orient = human.get_mesh_pos_orient(self.forearm)

        self.targets_pos_forearm_world = []
        for target_pos_on_arm, target in zip(self.targets_pos_on_forearm, self.targets_forearm):
            target_pos = np.array(p.multiplyTransforms(forearm_pos, forearm_orient, target_pos_on_arm, [0, 0, 0, 1], physicsClientId=env.id)[0])
            self.targets_pos_forearm_world.append(target_pos)
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




if __name__ == '__main__':

    dataset_info_dict = {}
    dataset_info_dict['gender'] = 'female'
    dataset_info_dict['data_ct'] = [1501, 2000, 2109]
    dataset_info_dict['posture'] = 'general_supine'
    dataset_info_dict['set_num'] = 14
    human = h.HumanMesh(dataset_info_dict)


    robot_arm = 'left'
    robot=PR2(robot_arm, human_type='mesh')


    HMR = HumanMeshReaching()


    if False:
        HMR.update_mattress_mesh(dataset_info_dict)
        human.update_human_mesh()


    env = AssistiveEnv(robot = robot, human = human, task='bed_bathing', bed_type='pressuresim', render=True)
    env.reset()

    print('building assistive env')
    robot, human, furniture = env.build_assistive_env(furniture_type='bed_mesh', fixed_human_base=False)


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
    pose_seq = [(pos + np.array([0.2, -0.2, -0.2]), orient)]
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
        env.take_step(joint_action, gains=env.config('robot_gains'), forces=env.config('robot_forces'))
        pos, orient = robot.get_pos_orient(robot.left_end_effector)
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

