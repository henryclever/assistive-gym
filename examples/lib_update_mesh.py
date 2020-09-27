import numpy as np
import time as time


class LibUpdateMesh():
    def __init__(self, dataset_info_dict, volatile_directory):
        self.posture = dataset_info_dict['posture']
        self.gender = dataset_info_dict['gender']
        self.data_ct_idx = dataset_info_dict['data_ct'][0]
        self.data_ct_l = dataset_info_dict['data_ct'][1]
        self.data_ct_h = dataset_info_dict['data_ct'][2]
        self.set_num = dataset_info_dict['set_num']
        self.volatile_directory = volatile_directory


    def update_gt_human_mesh(self):

        mesh_data_folder = "/home/henry/data_BR/resting_meshes/"+self.posture+"/roll0_"+self.gender[0]+"_lay_set"+str(self.set_num)+"_"+str(self.data_ct_l)+"_of_"+str(self.data_ct_h)+"_none_stiff/"
        hv = np.load(mesh_data_folder+"hv.npy")#, allow_pickle = True, encoding='latin1')
        hf = np.load(mesh_data_folder+"hf.npy")#, allow_pickle = True, encoding='latin1')

        for sample_idx in range(self.data_ct_idx,  self.data_ct_idx+1):# (0, np.shape(mv)[0]):
            human_verts = np.array(hv[sample_idx, :, :])/2.58872 #np.load(folder + "human_mesh_verts_py.npy")/2.58872
            human_verts = np.concatenate((human_verts[:, 2:3],human_verts[:, 0:1],human_verts[:, 1:2]), axis = 1)

            human_faces = np.array(hf[sample_idx, :, :]) #np.load(folder + "human_mesh_faces_py.npy")
            human_faces = np.concatenate((np.array([[0, 1, 2], [0, 4, 1], [0, 5, 4], [0, 2, 132], [0, 235, 5], [0, 132, 235] ]), human_faces), axis = 0)
            human_vf = [human_verts, human_faces]

        outmesh_human_path = self.volatile_directory+"human.obj"
        with open(outmesh_human_path, 'w') as fp:
            for v_idx in range(human_verts.shape[0]):
                fp.write('v %f %f %f\n' % (human_verts[v_idx, 0], human_verts[v_idx, 1], human_verts[v_idx, 2]))

            for f_idx in range(human_faces.shape[0]):
                fp.write('f %d %d %d\n' % (human_faces[f_idx, 0]+1, human_faces[f_idx, 1]+1, human_faces[f_idx, 2]+1))

    def update_est_human_mesh(self, m, root_shift):
        print(m.r[0],'from DF file,  shift: ', root_shift)
        human_verts = np.array(m.r) + root_shift#np.load(folder + "human_mesh_verts_py.npy")/2.58872
        #human_verts = np.concatenate((human_verts[:, 2:3],human_verts[:, 0:1],human_verts[:, 1:2]), axis = 1)

        human_faces = np.array(m.f) #np.load(folder + "human_mesh_faces_py.npy")
        human_faces = np.concatenate((np.array([[0, 1, 2], [0, 4, 1], [0, 5, 4], [0, 2, 132], [0, 235, 5], [0, 132, 235] ]), human_faces), axis = 0)
        human_vf = [human_verts, human_faces]

        outmesh_human_path = self.volatile_directory+"human_est.obj"
        with open(outmesh_human_path, 'w') as fp:
            for v_idx in range(human_verts.shape[0]):
                fp.write('v %f %f %f\n' % (human_verts[v_idx, 0], human_verts[v_idx, 1], human_verts[v_idx, 2]))

            for f_idx in range(human_faces.shape[0]):
                fp.write('f %d %d %d\n' % (human_faces[f_idx, 0]+1, human_faces[f_idx, 1]+1, human_faces[f_idx, 2]+1))




    def update_mattress_mesh(self):
        print("****Updating Mattress and Pressure Mat Mesh****")
        mesh_data_folder = "/home/henry/data_BR/resting_meshes/"+self.posture+\
                           "/roll0_"+self.gender[0]+\
                           "_lay_set"+str(self.set_num)+\
                           "_"+str(self.data_ct_l)+\
                           "_of_"+str(self.data_ct_h)+"_none_stiff/"

        mv = np.load(mesh_data_folder + "mv.npy")  # , allow_pickle = True, encoding='latin1')
        mf = np.load(mesh_data_folder + "mf.npy")  # , allow_pickle = True, encoding='latin1')
        bv = np.load(mesh_data_folder + "bv.npy", allow_pickle=True, encoding='latin1')
        bf = np.load(mesh_data_folder + "bf.npy", allow_pickle=True, encoding='latin1')

        for sample_idx in range(self.data_ct_idx,  self.data_ct_idx+1):  # (0, np.shape(mv)[0]):
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

        outmesh_mattress_path = self.volatile_directory+"bed_mattress.obj"
        with open(outmesh_mattress_path, 'w') as fp:
            for v_idx in range(mattress_verts.shape[0]):
                fp.write('v %f %f %f\n' % (mattress_verts[v_idx, 0], mattress_verts[v_idx, 1], mattress_verts[v_idx, 2]))

            for f_idx in range(mattress_faces.shape[0]):
                fp.write('f %d %d %d\n' % (
                mattress_faces[f_idx, 0] + 1, mattress_faces[f_idx, 1] + 1, mattress_faces[f_idx, 2] + 1))

        outmesh_pmat_path = self.volatile_directory+"bed_pmat.obj"
        with open(outmesh_pmat_path, 'w') as fp:
            for v_idx in range(pmat_verts.shape[0]):
                fp.write('v %f %f %f\n' % (pmat_verts[v_idx, 0], pmat_verts[v_idx, 1], pmat_verts[v_idx, 2]))

            for f_idx in range(pmat_faces.shape[0]):
                fp.write('f %d %d %d\n' % (pmat_faces[f_idx, 0] + 1, pmat_faces[f_idx, 1] + 1, pmat_faces[f_idx, 2] + 1))

