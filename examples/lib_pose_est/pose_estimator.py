#!/usr/bin/env python

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import *

# PyTorch libraries
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable



from smpl.smpl_webuser_py3.serialization import load_model
import chumpy as ch
# some_file.py
import sys

sys.path.insert(0, "../examples/lib_pose_est")


import convnet_br as convnet
# import tf.transformations as tft

# import hrl_lib.util as ut
import pickle as pickle


# from hrl_lib.util import load_pickle
def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f, encoding='latin1')


# Pose Estimation Libraries
from preprocessing_lib_ag import PreprocessingLib
from tensorprep_lib_ag import TensorPrepLib
from unpack_batch_lib_ag import UnpackBatchLib

import random
from scipy import ndimage
import scipy.stats as ss
from scipy.ndimage.interpolation import zoom
np.set_printoptions(threshold=sys.maxsize)

MAT_WIDTH = 0.762  # metres
MAT_HEIGHT = 1.854  # metres
MAT_HALF_WIDTH = MAT_WIDTH / 2
NUMOFTAXELS_X = 64  # 73 #taxels
NUMOFTAXELS_Y = 27  # 30
NUMOFOUTPUTDIMS = 3
NUMOFOUTPUTNODES_TEST = 24
INTER_SENSOR_DISTANCE = 0.0286  # metres
LOW_TAXEL_THRESH_X = 0
LOW_TAXEL_THRESH_Y = 0
HIGH_TAXEL_THRESH_X = (NUMOFTAXELS_X - 1)
HIGH_TAXEL_THRESH_Y = (NUMOFTAXELS_Y - 1)
TEST_SUBJECT = 9
CAM_BED_DIST = 1.66
DEVICE = 1

torch.set_num_threads(1)
if False:#torch.cuda.is_available():
    # Use for GPU
    GPU = True
    dtype = torch.cuda.FloatTensor
    torch.cuda.set_device(DEVICE)
    print('######################### CUDA is available! #############################')
else:
    # Use for CPU
    GPU = False
    dtype = torch.FloatTensor
    print('############################## USING CPU #################################')


class PoseEstimator():
    '''Gets the dictionary of pressure maps from the training database,
    and will have API to do all sorts of training with it.'''



    def __init__(self, dataset_info_dict):
        self.gender = dataset_info_dict['gender']
        model_path = '../smpl/models/basicModel_' + self.gender[0] + '_lbs_10_207_0_v1.0.0.pkl'
        m = load_model(model_path)

        self.m = m


    def init(self, dataset_info_dict):


        self.posture = dataset_info_dict['posture']
        self.gender = dataset_info_dict['gender']
        self.data_ct_idx = dataset_info_dict['data_ct'][0]
        self.data_ct_l = dataset_info_dict['data_ct'][1]
        self.data_ct_h = dataset_info_dict['data_ct'][2]
        self.set_num = dataset_info_dict['set_num']

        self.FILEPATH_PREFIX = "../../../data_BR"


        self.NETWORK_1 = "resnet50_1_anglesDC_184000ct_128b_x1pm_tnh_clns20p_100e_0.0001lr"
        self.NETWORK_2 = "resnet50_2_anglesDC_184000ct_128b_x1pm_0.5rtojtdpth_depthestin_angleadj_tnh_clns20p_100e_0.0001lr"

        item_test = [self.gender[0], self.posture+"/","train_roll0_"+self.gender[0]+"_lay_set10to13_8000"]


        PARTITION = item_test[1]
        TESTING_FILENAME = item_test[2]

        testing_database_file_f = []
        testing_database_file_m = [] #141 total training loss at epoch 9



        if self.gender[0] == "f":
            testing_database_file_f.append(self.FILEPATH_PREFIX+'/synth/'+PARTITION+TESTING_FILENAME+'.p')
        else:
            testing_database_file_m.append(self.FILEPATH_PREFIX+'/synth/'+PARTITION+TESTING_FILENAME+'.p')



        # change this to 'direct' when you are doing baseline methods

        self.CTRL_PNL = {}
        self.CTRL_PNL['loss_vector_type'] = 'anglesDC'
        self.CTRL_PNL['CNN'] = 'resnet'

        self.CTRL_PNL['verbose'] = False
        self.CTRL_PNL['pmr'] = True
        self.CTRL_PNL['batch_size'] = 1
        self.CTRL_PNL['num_epochs'] = 100
        self.CTRL_PNL['incl_inter'] = True
        self.CTRL_PNL['shuffle'] = False
        self.CTRL_PNL['incl_pmat_cntct_input'] = True
        self.CTRL_PNL['half_network_size'] = False
        self.CTRL_PNL['double_network_size'] = False
        self.CTRL_PNL['incl_ht_wt_channels'] = False
        self.CTRL_PNL['loss_root'] = False
        self.CTRL_PNL['omit_cntct_sobel'] = False
        self.CTRL_PNL['omit_pimg_cntct_sobel'] = False
        self.CTRL_PNL['dropout'] = False
        self.CTRL_PNL['lock_root'] = False
        self.CTRL_PNL['num_input_channels'] = 2
        self.CTRL_PNL['GPU'] = GPU
        self.CTRL_PNL['dtype'] = dtype
        repeat_real_data_ct = 3
        self.CTRL_PNL['regr_angles'] = False
        self.CTRL_PNL['recon_map_labels'] = False #can only be true if we have 100% synthetic data for training
        self.CTRL_PNL['recon_map_labels_test'] = False #can only be true is we have 100% synth for testing
        self.CTRL_PNL['recon_map_output'] = True #self.CTRL_PNL['recon_map_labels']
        self.CTRL_PNL['recon_map_input_est'] = False  #do this if we're working in a two-part regression
        self.CTRL_PNL['adjust_ang_from_est'] = self.CTRL_PNL['recon_map_input_est'] #holds betas and root same as prior estimate
        self.CTRL_PNL['clip_sobel'] = True
        self.CTRL_PNL['clip_betas'] = True
        self.CTRL_PNL['mesh_bottom_dist'] = True
        self.CTRL_PNL['full_body_rot'] = True
        self.CTRL_PNL['normalize_per_image'] = True
        if self.CTRL_PNL['normalize_per_image'] == False:
            self.CTRL_PNL['normalize_std'] = True
        else:
            self.CTRL_PNL['normalize_std'] = False
        self.CTRL_PNL['all_tanh_activ'] = True
        self.CTRL_PNL['L2_contact'] = True
        self.CTRL_PNL['pmat_mult'] = int(1)
        self.CTRL_PNL['cal_noise'] = True
        self.CTRL_PNL['cal_noise_amt'] = 0.1
        self.CTRL_PNL['output_only_prev_est'] = False
        self.CTRL_PNL['first_pass'] = True
        self.CTRL_PNL['align_procr'] = False
        self.CTRL_PNL['depth_in'] = False
        self.CTRL_PNL['depth_out_unet'] = False



        if self.CTRL_PNL['cal_noise'] == True:
            self.CTRL_PNL['incl_pmat_cntct_input'] = False #if there's calibration noise we need to recompute this every batch
            self.CTRL_PNL['clip_sobel'] = False

        if self.CTRL_PNL['incl_pmat_cntct_input'] == True:
            self.CTRL_PNL['num_input_channels'] += 1
        if self.CTRL_PNL['recon_map_input_est'] == True: #for a two part regression
            self.CTRL_PNL['num_input_channels'] += 2
        self.CTRL_PNL['num_input_channels_batch0'] = np.copy(self.CTRL_PNL['num_input_channels'])
        if self.CTRL_PNL['incl_ht_wt_channels'] == True:
            self.CTRL_PNL['num_input_channels'] += 2
        if self.CTRL_PNL['cal_noise'] == True:
            self.CTRL_PNL['num_input_channels'] += 1
        if self.CTRL_PNL['depth_in'] == True:
            self.CTRL_PNL['num_input_channels'] += 1


        pmat_std_from_mult = ['N/A', 11.70153502792190, 19.90905848383454, 23.07018866032369, 0.0, 25.50538629767412]
        if self.CTRL_PNL['cal_noise'] == False:
            sobel_std_from_mult = ['N/A', 29.80360490415032, 33.33532963163579, 34.14427844692501, 0.0, 34.86393494050921]
        else:
            sobel_std_from_mult = ['N/A', 45.61635847182483, 77.74920396659292, 88.89398421073700, 0.0, 97.90075708182506]

        self.CTRL_PNL['norm_std_coeffs'] =  [1./41.80684362163343,  #contact
                                             1./45.08513083167194,  #neg est depth
                                             1./43.55800622930469,  #cm est
                                             1./pmat_std_from_mult[int(self.CTRL_PNL['pmat_mult'])], #pmat x5
                                             1./sobel_std_from_mult[int(self.CTRL_PNL['pmat_mult'])], #pmat sobel
                                             1./1.0,                #OUTPUT DO NOTHING
                                             1./1.0,                #OUTPUT DO NOTHING
                                             1. / 30.216647403350,  #weight
                                             1. / 14.629298141231]  #height


        if self.CTRL_PNL['normalize_std'] == False:
            for i in range(9):
                self.CTRL_PNL['norm_std_coeffs'][i] *= 0.
                self.CTRL_PNL['norm_std_coeffs'][i] += 1.


        self.CTRL_PNL['convnet_fp_prefix'] = '../data_BR/convnets/'

        if self.CTRL_PNL['recon_map_output'] == True: #we need all the vertices if we're going to regress the depth maps
            self.verts_list = "all"
        else:
            self.verts_list = [1325, 336, 1032, 4515, 1374, 4848, 1739, 5209, 1960, 5423]

        # Entire pressure dataset with coordinates in world frame



        self.mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)
        self.output_size_test = (NUMOFOUTPUTNODES_TEST, NUMOFOUTPUTDIMS)
        self.parents = np.array([4294967295, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]).astype(np.int32)



        #################################### PREP TESTING DATA ##########################################
        # load in the test file

        test_dat_f_synth = TensorPrepLib().load_files_to_database(testing_database_file_f, creation_type = 'synth', reduce_data = False, test = True)
        test_dat_m_synth = TensorPrepLib().load_files_to_database(testing_database_file_m, creation_type = 'synth', reduce_data = False, test = True)
        test_dat_f_real = TensorPrepLib().load_files_to_database(testing_database_file_f, creation_type = 'real', reduce_data = False, test = True)
        test_dat_m_real = TensorPrepLib().load_files_to_database(testing_database_file_m, creation_type = 'real', reduce_data = False, test = True)

        self.test_x_flat = []  # Initialize the testing pressure mat list
        self.test_x_flat = TensorPrepLib().prep_images(self.test_x_flat, test_dat_f_synth, test_dat_m_synth, num_repeats = 1)
        self.test_x_flat = list(np.clip(np.array(self.test_x_flat) * float(self.CTRL_PNL['pmat_mult']), a_min=0, a_max=100))
        self.test_x_flat = TensorPrepLib().prep_images(self.test_x_flat, test_dat_f_real, test_dat_m_real, num_repeats = 1)

        if self.CTRL_PNL['cal_noise'] == False:
            self.test_x_flat = PreprocessingLib().preprocessing_blur_images(self.test_x_flat, self.mat_size, sigma=0.5)

        if len(self.test_x_flat) == 0: print("NO TESTING DATA INCLUDED")

        if self.CTRL_PNL['recon_map_labels_test'] == True:
            self.mesh_reconstruction_maps = [] #Initialize the precomputed depth and contact maps. only synth has this label.
            self.mesh_reconstruction_maps = TensorPrepLib().prep_reconstruction_gt(self.mesh_reconstruction_maps, test_dat_f_synth, test_dat_m_synth, num_repeats = 1)

        else:
            self.mesh_reconstruction_maps = None

        if self.CTRL_PNL['recon_map_input_est'] == True:
            self.reconstruction_maps_input_est = [] #Initialize the precomputed depth and contact map input estimates
            self.reconstruction_maps_input_est = TensorPrepLib().prep_reconstruction_input_est(self.reconstruction_maps_input_est,
                                                                                             test_dat_f_synth, test_dat_m_synth, num_repeats = 1)
        else:
            self.reconstruction_maps_input_est = None


        if self.CTRL_PNL['depth_in'] == True:
            self.depth_images = []
            self.depth_images = TensorPrepLib().prep_depth_input_images(self.depth_images, test_dat_f_synth, test_dat_m_synth, num_repeats = 1)
        else:
            self.depth_images = None

        if self.CTRL_PNL['depth_out_unet'] == True:
            self.depth_images_out_unet = []
            self.depth_images_out_unet = TensorPrepLib().prep_depth_input_images(self.depth_images_out_unet, test_dat_f_synth, test_dat_m_synth, num_repeats=1, depth_type = 'human_only')
        else:
            self.depth_images_out_unet = None


        test_xa = PreprocessingLib().preprocessing_create_pressure_angle_stack(self.test_x_flat,
                                                                                self.mat_size,
                                                                                self.CTRL_PNL)

        test_xa = TensorPrepLib().append_trainxa_besides_pmat_edges(np.array(test_xa),
                                                              CTRL_PNL = self.CTRL_PNL,
                                                              mesh_reconstruction_maps_input_est = self.reconstruction_maps_input_est,
                                                              mesh_reconstruction_maps = self.mesh_reconstruction_maps,
                                                              depth_images = self.depth_images,
                                                              depth_images_out_unet = self.depth_images_out_unet)


        #normalize the input
        if self.CTRL_PNL['normalize_std'] == True:
            test_xa = TensorPrepLib().normalize_network_input(test_xa, self.CTRL_PNL)

        self.test_x_tensor = torch.Tensor(test_xa)

        test_y_flat = []  # Initialize the ground truth listhave

        test_y_flat = TensorPrepLib().prep_labels(test_y_flat, test_dat_f_synth, num_repeats = 1,
                                                    z_adj = -0.075, gender = "f", is_synth = True,
                                                    loss_vector_type = self.CTRL_PNL['loss_vector_type'],
                                                    initial_angle_est = self.CTRL_PNL['adjust_ang_from_est'],
                                                    full_body_rot = self.CTRL_PNL['full_body_rot'])
        test_y_flat = TensorPrepLib().prep_labels(test_y_flat, test_dat_m_synth, num_repeats = 1,
                                                    z_adj = -0.075, gender = "m", is_synth = True,
                                                    loss_vector_type = self.CTRL_PNL['loss_vector_type'],
                                                    initial_angle_est = self.CTRL_PNL['adjust_ang_from_est'],
                                                    full_body_rot = self.CTRL_PNL['full_body_rot'])

        test_y_flat = TensorPrepLib().prep_labels(test_y_flat, test_dat_f_real, num_repeats = 1,
                                                    z_adj = 0.0, gender = "f", is_synth = False,
                                                    loss_vector_type = self.CTRL_PNL['loss_vector_type'],
                                                    initial_angle_est = self.CTRL_PNL['adjust_ang_from_est'])
        test_y_flat = TensorPrepLib().prep_labels(test_y_flat, test_dat_m_real, num_repeats = 1,
                                                    z_adj = 0.0, gender = "m", is_synth = False,
                                                    loss_vector_type = self.CTRL_PNL['loss_vector_type'],
                                                    initial_angle_est = self.CTRL_PNL['adjust_ang_from_est'])

        if self.CTRL_PNL['normalize_std'] == True:
            test_y_flat = TensorPrepLib().normalize_wt_ht(test_y_flat, self.CTRL_PNL)

        self.test_y_tensor = torch.Tensor(test_y_flat)


        #print(self.test_x_tensor.shape, 'Input testing tensor shape')
        #print(self.test_y_tensor.shape, 'Output testing tensor shape')


        self.init_convnet_test()




    def init_convnet_test(self):

        #self.test_x_tensor = self.test_x_tensor.unsqueeze(1)
        self.test_dataset = torch.utils.data.TensorDataset(self.test_x_tensor, self.test_y_tensor)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, self.CTRL_PNL['batch_size'], shuffle=self.CTRL_PNL['shuffle'])


        fc_output_size = 85## 10 + 3 + 24*3 --- betas, root shift, rotations

        if self.CTRL_PNL['full_body_rot'] == True:
            fc_output_size += 3



        self.model = torch.load(self.FILEPATH_PREFIX + "/convnets/"+self.NETWORK_1+".pt", map_location='cpu', encoding='latin1')
        self.model2 = torch.load(self.FILEPATH_PREFIX + "/convnets/"+self.NETWORK_2+".pt", map_location='cpu', encoding='latin1')
        


        #self.model2 = None
        pp = 0
        for p in list(self.model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        #print('LOADED. num params: ', pp)


        # Run model on GPU if available
        #if torch.cuda.is_available():
        if GPU == True:
            self.model = self.model.cuda()
            if self.model2 is not None:
                self.model2 = self.model2.cuda()

        self.model = self.model.eval()
        if self.model2 is not None:
            self.model2 = self.model2.eval()



    def estimate_pose(self):

        '''
        Train the model for one epoch.
        '''
        # Some models use slightly different forward passes and train and test
        # time (e.g., any model with Dropout). This puts the model in train mode
        # (as opposed to eval mode) so it knows which one to use.

        model_path = '../smpl/models/basicModel_' + self.gender[0] + '_lbs_10_207_0_v1.0.0.pkl'
        m = load_model(model_path)

        RESULTS_DICT = {}
        RESULTS_DICT['j_err'] = []
        RESULTS_DICT['betas'] = []
        RESULTS_DICT['dir_v_err'] = []
        RESULTS_DICT['v2v_err'] = []
        RESULTS_DICT['dir_v_limb_err'] = []
        RESULTS_DICT['v_to_gt_err'] = []
        RESULTS_DICT['v_limb_to_gt_err'] = []
        RESULTS_DICT['gt_to_v_err'] = []
        RESULTS_DICT['precision'] = []
        RESULTS_DICT['recall'] = []
        RESULTS_DICT['overlap_d_err'] = []
        RESULTS_DICT['all_d_err'] = []
        init_time = time.time()

        with torch.autograd.set_detect_anomaly(True):

            # This will loop a total = training_images/batch_size times
            for batch_idx, batch in enumerate(self.test_loader):
                if batch_idx != self.data_ct_idx:
                    continue

                batch1 = batch[1].clone()

                betas_gt = torch.mean(batch[1][:, 72:82], dim = 0).numpy()
                angles_gt = torch.mean(batch[1][:, 82:154], dim = 0).numpy()
                root_shift_est_gt = torch.mean(batch[1][:, 154:157], dim = 0).numpy()

                NUMOFOUTPUTDIMS = 3
                NUMOFOUTPUTNODES_TEST = 24
                self.output_size_test = (NUMOFOUTPUTNODES_TEST, NUMOFOUTPUTDIMS)

                self.CTRL_PNL['adjust_ang_from_est'] = False
                self.CTRL_PNL['recon_map_labels'] = False

                #print(self.CTRL_PNL['num_input_channels_batch0'], batch[0].size(), 'batch[0] for mod1')


                if self.CTRL_PNL['depth_in'] == True:
                    depth_quad = batch[0][:, 2:, :, :].clone()

                self.CTRL_PNL['align_procr'] = False
                scores, INPUT_DICT, OUTPUT_DICT = UnpackBatchLib().unpack_batch(batch, False, self.model,
                                                                                            self.CTRL_PNL)


                mdm_est_pos = OUTPUT_DICT['batch_mdm_est'].clone().unsqueeze(1)  # / 16.69545796387731
                mdm_est_neg = OUTPUT_DICT['batch_mdm_est'].clone().unsqueeze(1)  # / 45.08513083167194
                mdm_est_pos[mdm_est_pos < 0] = 0
                mdm_est_neg[mdm_est_neg > 0] = 0
                mdm_est_neg *= -1
                cm_est = OUTPUT_DICT['batch_cm_est'].clone().unsqueeze(1) * 100  # / 43.55800622930469

                # 1. / 16.69545796387731,  # pos est depth
                # 1. / 45.08513083167194,  # neg est depth
                # 1. / 43.55800622930469,  # cm est

                sc_sample1 = OUTPUT_DICT['batch_targets_est'].clone()
                sc_sample1 = sc_sample1[0, :].squeeze() / 1000
                sc_sample1 = sc_sample1.view(self.output_size_test)
                # print sc_sample1

                if self.model2 is not None:
                    print("Using model 2")
                    batch_cor = []

                    if self.CTRL_PNL['cal_noise'] == False:
                        batch_cor.append(torch.cat((batch[0][:, 0:1, :, :],
                                                    mdm_est_neg.type(torch.FloatTensor),
                                                    cm_est.type(torch.FloatTensor),
                                                    batch[0][:, 1:, :, :]), dim=1))
                    else:
                        if self.CTRL_PNL['pmr'] == True:
                            batch_cor.append(torch.cat((mdm_est_neg.type(torch.FloatTensor),
                                                        cm_est.type(torch.FloatTensor),
                                                        batch[0][:, 0:, :, :]), dim=1))
                        else:
                            batch_cor.append(batch[0])

                    if self.CTRL_PNL['depth_in'] == True:
                        batch_cor[0] = torch.cat((batch_cor[0], depth_quad), dim = 1)

                    if self.CTRL_PNL['full_body_rot'] == False:
                        batch_cor.append(torch.cat((batch1,
                                                    OUTPUT_DICT['batch_betas_est'].cpu(),
                                                    OUTPUT_DICT['batch_angles_est'].cpu(),
                                                    OUTPUT_DICT['batch_root_xyz_est'].cpu()), dim=1))
                    elif self.CTRL_PNL['full_body_rot'] == True:
                        batch_cor.append(torch.cat((batch1,
                                                    OUTPUT_DICT['batch_betas_est'].cpu(),
                                                    OUTPUT_DICT['batch_angles_est'].cpu(),
                                                    OUTPUT_DICT['batch_root_xyz_est'].cpu(),
                                                    OUTPUT_DICT['batch_root_atan2_est'].cpu()), dim=1))

                    self.CTRL_PNL['adjust_ang_from_est'] = True

                    if self.CTRL_PNL['pmr'] == True:
                        self.CTRL_PNL['num_input_channels_batch0'] += 2

                    #print(self.CTRL_PNL['num_input_channels_batch0'], batch_cor[0].size(), 'batch[0] for mod2')
                    self.CTRL_PNL['align_procr'] = False

                    scores, INPUT_DICT, OUTPUT_DICT = UnpackBatchLib().unpack_batch(batch_cor, is_training=False,
                                                                                    model=self.model2,
                                                                                    CTRL_PNL=self.CTRL_PNL)
                    if self.CTRL_PNL['pmr'] == True:
                        self.CTRL_PNL['num_input_channels_batch0'] -= 2


                self.CTRL_PNL['first_pass'] = False



                q = OUTPUT_DICT['batch_mdm_est'].data.cpu().numpy().reshape(OUTPUT_DICT['batch_mdm_est'].size()[0], 64, 27) * -1
                q = np.mean(q, axis=0)

                camera_point = [1.09898028, 0.46441343, -CAM_BED_DIST]


                RESULTS_DICT['betas'].append(OUTPUT_DICT['batch_betas_est_post_clip'].cpu().numpy()[0])
                #print(RESULTS_DICT['betas'][-1], "BETAS")





                #do 3D viz
                pmat = batch[0][0, 1, :, :].clone().numpy() * 25.50538629767412
                # print pmat.shape

                for beta in range(betas_gt.shape[0]):
                    m.betas[beta] = betas_gt[beta]
                for angle in range(angles_gt.shape[0]):
                    m.pose[angle] = angles_gt[angle]




                bed_leg_ht = 0.3048
                mattress_ht = 0.2032
                entire_bed_shift = np.array([-0.45775, -0.98504, bed_leg_ht+mattress_ht])


                self.root_shift_gt = np.array(OUTPUT_DICT['root_shift_gt'] - m.J[0, :])
                #self.smpl_verts = np.array(m.r) + self.root_shift_gt + entire_bed_shift #this is for ground truth
                self.smpl_verts = OUTPUT_DICT['verts'][0] + entire_bed_shift #this is for estimate


                self.joint_locs_trans_abs = []
                for i in range(24):

                    #to_append = list(np.array(m.J_transformed[i, :])+ self.root_shift_gt + entire_bed_shift) #this is for ground truth
                    to_append = list(np.array(OUTPUT_DICT['targets_est_np'][0][i]) + entire_bed_shift) #this is for estimate
                    self.joint_locs_trans_abs.append(to_append)

                self.m = m


        return self.m, self.smpl_verts, self.joint_locs_trans_abs



    def update_est_human_mesh(self, m, smpl_verts):

        legacy_x_shift = -0.286 + 0.0143
        legacy_y_shift = -0.286 + 0.0143
        pmat_ht = 0.075

        smpl_verts += np.array([legacy_x_shift, legacy_y_shift, pmat_ht])#np.array(root_shift)

        smpl_faces = np.array(m.f)

        outmesh_human_path = "/home/henry/git/assistive-gym/assistive_gym/envs/assets/human_mesh/human_est.obj"
        with open(outmesh_human_path, 'w') as fp:
            for v_idx in range(smpl_verts.shape[0]):
                fp.write('v %f %f %f\n' % (smpl_verts[v_idx, 0], smpl_verts[v_idx, 1], smpl_verts[v_idx, 2]))

            for f_idx in range(smpl_faces.shape[0]):
                fp.write('f %d %d %d\n' % (smpl_faces[f_idx, 0]+1, smpl_faces[f_idx, 1]+1, smpl_faces[f_idx, 2]+1))


