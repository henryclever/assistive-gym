import numpy as np
import time as time
from lib_pose_est.kinematics_lib_ag import KinematicsLib









def append_from_index(pose_sequence, pos_goal, euler_angles_goal, pos_add):
    R = KinematicsLib().eulerAnglesToRotationMatrix(euler_angles_goal)
    orient = KinematicsLib().eulerAnglesToQuaternion(euler_angles_goal)

    print(R)

    rotated_point = np.matmul(R, np.array([[pos_add[2]], [pos_add[1]], [pos_add[0]]])).T[0]
    #rotated_point = -np.matmul(R, np.array([pos_add[0], pos_add[1], pos_add[2]]))
    rotated_point = np.array([rotated_point[2], -rotated_point[1], -rotated_point[0]])

    print(pos_add, rotated_point)


    pose_sequence.append((pos_goal + rotated_point, orient))



if __name__ == '__main__':

    vectors = [[1,1,0],[-1,-1,0],[-1,1,0],[1,-1,0],
               [0,1,1],[0,-1,-1],[0,-1,1],[0,1,-1],
               [1,0,1],[-1,0,-1],[-1,0,1],[1,0,-1]]
    vectors = [[1, 1, 1]]
    #vectors = [[0.854, 0.34, 0.22]]

    #updated_euler_angles = KinematicsLib().vectorToEulerAngles(np.array(vectors[4]))

    for vector in vectors:
        vector= np.array(vector)/np.linalg.norm(vector)

        updated_euler_angles = KinematicsLib().vectorToEulerAngles(np.array(vector))


        radius = 0.1
        pose_seq = []
        pos_goal = np.array([0.0, 0.0, 0.0])
        print(vector)
        for theta in np.linspace(0, 4 * np.pi, 100):
            append_from_index(pose_seq, pos_goal, updated_euler_angles, np.array([radius * np.cos(theta), radius * np.sin(theta), 0]))
            break


    '''for i in range(12):
        vectors[i] = vectors[i]/np.linalg.norm(vectors[i])


    angle_H = []
    for vector in vectors:
        angle_H.append(np.pi+np.arctan2(vector[1], vector[2]))
        if angle_H[-1] > np.pi:
            angle_H[-1] -= 2*np.pi



    anglePITCH = []
    for vector in vectors:
        anglePITCH.append(np.arcsin(-vector[0]))


    print("H", angle_H)
    print("pitch", anglePITCH)

    #for i in range(0,1):
    #for i in range(4,5):
    for i in range(0, 12):
        #for j in range(-85, -70):
        #for j in range(-10, 10):
        angle_possib = []
        vector_possib = []
        error = []
        for j in range(-314, 314):
            radians = float(j/100)

            angle_possib.append(radians)
            #print('eulers', [radians, anglePITCH[i], angle_H[i]])
            vect = KinematicsLib().eulerAnglesToRotationMatrix([radians, anglePITCH[i], angle_H[i]])
            orient = KinematicsLib().eulerAnglesToQuaternion([radians, anglePITCH[i], angle_H[i]])
            #print("vector", vect, vectors[i])
            vector_possib.append([vect[0,2], vect[0,1], -vect[0,0]])

            error.append(np.linalg.norm(vector_possib[-1]- vectors[i]))

            #print("error", np.linalg.norm(vector_possib[-1]- vectors[i]), angle_possib[-1])


        min_idx = np.argmin(error)
        angle_B = angle_possib[min_idx]
        print(angle_B)'''






    '''angle_H = []
    for vector in vectors:
        angle_H.append(np.arctan2(vector[1], vector[0]))
        if angle_H[-1] > np.pi:
            angle_H[-1] -= 2*np.pi


    anglePITCH = []
    for vector in vectors:
        anglePITCH.append(np.arcsin(vector[2]))

    U = []
    for vector in vectors:
        U.append([vector[1], vector[2], vector[0]])

    W0 = []
    for vector in vectors:
        W0.append([-vector[1], vector[0], 0])

    U0 = []
    for i in range(12):
        U0.append(np.cross(W0[i], vectors[i]))

    angle_B = []
    for i in range(12):
        angle_B.append(np.arctan2( np.dot(U0[i], U[i])/np.linalg.norm(U0[i]), np.dot(W0[i], U[i])/np.linalg.norm(W0[i])))


    for i in range(12):
        #print(vectors[i], [angle_H[i], anglePITCH[i], angle_B[i]])
        R = KinematicsLib().eulerAnglesToRotationMatrix([angle_H[i], anglePITCH[i], angle_B[i]])
        print([R[2,2], R[2,1], R[2,0]])'''



    '''Wo = []
    for vector in vectors:
        type_list = []
        type_list.append([vector[0], vector[1], 0])
        type_list.append([-vector[0], vector[1], 0])
        type_list.append([vector[0], -vector[1], 0])
        type_list.append([-vector[0], -vector[1], 0])
        type_list.append([vector[1], vector[0], 0])
        type_list.append([-vector[1], vector[0], 0])
        type_list.append([vector[1], -vector[0], 0])
        type_list.append([-vector[1], -vector[0], 0])
        type_list.append([vector[0], 0, vector[2]])
        type_list.append([-vector[0], 0, vector[2]])
        type_list.append([vector[0], 0, -vector[2]])
        type_list.append([-vector[0], 0, -vector[2]])
        type_list.append([vector[2], 0, vector[0]])
        type_list.append([-vector[2], 0, vector[0]])
        type_list.append([vector[2], 0, -vector[0]])
        type_list.append([-vector[2], 0, -vector[0]])
        type_list.append([0, vector[1], vector[2]])
        type_list.append([0, -vector[1], vector[2]])
        type_list.append([0, vector[1], -vector[2]])
        type_list.append([0, -vector[1], -vector[2]])
        type_list.append([0, vector[2], vector[1]])
        type_list.append([0, -vector[2], vector[1]])
        type_list.append([0, vector[2], -vector[1]])
        type_list.append([0, -vector[2], -vector[1]])
        Wo.append(type_list)


    Uo = []
    for i in range(12):
        vector = vectors[i]
        Wo_type_list = Wo[i]
        Uo_type_list_cross = []
        for item in Wo_type_list:
            #Uo_type_list_cross.append(np.cross(vector, item))
            Uo_type_list_cross.append(np.cross(item, vector))


            #print item, Wo_type_list_cross[-2], Wo_type_list_cross[-1]


        Uo.append(Uo_type_list_cross)
        #print Wo_type_list, 'Wo type list'
        #print Wo_type_list_cross, 'cross'
        #break

    U = []
    for i in range(12):
        vector = vectors[i]
        U_sublist = []

        U_sublist.append([vector[0], vector[1], vector[2]])
        U_sublist.append([-vector[0], vector[1], vector[2]])
        U_sublist.append([vector[0], -vector[1], vector[2]])
        U_sublist.append([vector[0], vector[1], -vector[2]])
        U_sublist.append([-vector[0], -vector[1], vector[2]])
        U_sublist.append([-vector[0], vector[1], -vector[2]])
        U_sublist.append([vector[0], -vector[1], -vector[2]])
        U_sublist.append([-vector[0], -vector[1], -vector[2]])

        U_sublist.append([vector[0], vector[2], vector[1]])
        U_sublist.append([-vector[0], vector[2], vector[1]])
        U_sublist.append([vector[0], -vector[2], vector[1]])
        U_sublist.append([vector[0], vector[2], -vector[1]])
        U_sublist.append([-vector[0], -vector[2], vector[1]])
        U_sublist.append([-vector[0], vector[2], -vector[1]])
        U_sublist.append([vector[0], -vector[2], -vector[1]])
        U_sublist.append([-vector[0], -vector[2], -vector[1]])

        U_sublist.append([vector[1], vector[0], vector[2]])
        U_sublist.append([-vector[1], vector[0], vector[2]])
        U_sublist.append([vector[1], -vector[0], vector[2]])
        U_sublist.append([vector[1], vector[0], -vector[2]])
        U_sublist.append([-vector[1], -vector[0], vector[2]])
        U_sublist.append([-vector[1], vector[0], -vector[2]])
        U_sublist.append([vector[1], -vector[0], -vector[2]])
        U_sublist.append([-vector[1], -vector[0], -vector[2]])

        U_sublist.append([vector[1], vector[2], vector[0]])
        U_sublist.append([-vector[1], vector[2], vector[0]])
        U_sublist.append([vector[1], -vector[2], vector[0]])
        U_sublist.append([vector[1], vector[2], -vector[0]])
        U_sublist.append([-vector[1], -vector[2], vector[0]])
        U_sublist.append([-vector[1], vector[2], -vector[0]])
        U_sublist.append([vector[1], -vector[2], -vector[0]])
        U_sublist.append([-vector[1], -vector[2], -vector[0]])

        U_sublist.append([vector[2], vector[1], vector[0]])
        U_sublist.append([-vector[2], vector[1], vector[0]])
        U_sublist.append([vector[2], -vector[1], vector[0]])
        U_sublist.append([vector[2], vector[1], -vector[0]])
        U_sublist.append([-vector[2], -vector[1], vector[0]])
        U_sublist.append([-vector[2], vector[1], -vector[0]])
        U_sublist.append([vector[2], -vector[1], -vector[0]])
        U_sublist.append([-vector[2], -vector[1], -vector[0]])

        U_sublist.append([vector[2], vector[0], vector[1]])
        U_sublist.append([-vector[2], vector[0], vector[1]])
        U_sublist.append([vector[2], -vector[0], vector[1]])
        U_sublist.append([vector[2], vector[0], -vector[1]])
        U_sublist.append([-vector[2], -vector[0], vector[1]])
        U_sublist.append([-vector[2], vector[0], -vector[1]])
        U_sublist.append([vector[2], -vector[0], -vector[1]])
        U_sublist.append([-vector[2], -vector[0], -vector[1]])

        U.append(U_sublist)

    roll_comp_1 = []
    roll_comp_2 = []
    for i in range(12):
        roll_comp_1_sublist = []
        roll_comp_2_sublist = []

        Wo_sublist = Wo[i]
        U_sublist = U[i]
        Uo_sublist = Uo[i]


        for Wo_sub_cand in Wo_sublist:
            for U_sub_cand in U_sublist:
                roll_comp_1_sublist.append(np.dot(Wo_sub_cand, U_sub_cand)/np.linalg.norm(Wo_sub_cand))

        for Uo_sub_cand in Uo_sublist:
            for U_sub_cand in U_sublist:
                roll_comp_2_sublist.append(np.dot(Uo_sub_cand, U_sub_cand)/np.linalg.norm(Uo_sub_cand))

        roll_comp_1.append(roll_comp_1_sublist)
        roll_comp_2.append(roll_comp_2_sublist)
        #print len(roll_comp_1_sublist), len(roll_comp_2_sublist)


    angle_B_list = []
    for i in range(12):
        atan_sublist = []
        for j in range(len(roll_comp_1[0])):
            atan_sublist.append(np.arctan2(roll_comp_1[i][j], roll_comp_2[i][j]))
            atan_sublist.append(np.arctan2(-roll_comp_1[i][j], roll_comp_2[i][j]))
            atan_sublist.append(np.arctan2(roll_comp_1[i][j], -roll_comp_2[i][j]))
            atan_sublist.append(np.arctan2(-roll_comp_1[i][j], -roll_comp_2[i][j]))
            atan_sublist.append(np.arctan2(roll_comp_2[i][j], roll_comp_1[i][j]))
            atan_sublist.append(np.arctan2(-roll_comp_2[i][j], roll_comp_1[i][j]))
            atan_sublist.append(np.arctan2(roll_comp_2[i][j], -roll_comp_1[i][j]))
            atan_sublist.append(np.arctan2(-roll_comp_2[i][j], -roll_comp_1[i][j]))
        angle_B_list.append(atan_sublist)
        #print len(roll_comp_1), len(atan_sublist)

    print len(angle_B_list[0])

    #print angle_B_list[0]

    ground_truth = [-np.pi/4, -np.pi/4, np.pi/4, np.pi/4, 0, 0, 0, 0,   0, np.pi, 0, np.pi]
    for j in range(len(roll_comp_1[0])):

        error = 0
        for i in range(12):
            error += np.abs(angle_B_list[i][j] - ground_truth[i])

        #print error
        if np.abs(error) < 9:
            print "error", error
            print "curr", angle_B_list[0][j], angle_B_list[1][j], angle_B_list[2][j], angle_B_list[3][j], angle_B_list[4][j], angle_B_list[5][j], \
                  angle_B_list[6][j], angle_B_list[7][j], angle_B_list[8][j], angle_B_list[9][j], angle_B_list[10][j], angle_B_list[11][j]
            print "gt", ground_truth

    for i in range(12):
        print vectors[i], angle_H[i], anglePITCH[i]'''
