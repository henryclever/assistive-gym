#!/usr/bin/env python
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import *


import random
from scipy import ndimage
import scipy.stats as ss
from scipy.ndimage.interpolation import zoom




# PyTorch libraries
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import transforms
from torch.autograd import Variable

MAT_WIDTH = 0.762 #metres
MAT_HEIGHT = 1.854 #metres
MAT_HALF_WIDTH = MAT_WIDTH/2
NUMOFTAXELS_X = 84#73 #taxels
NUMOFTAXELS_Y = 47#30
INTER_SENSOR_DISTANCE = 0.0286#metres
LOW_TAXEL_THRESH_X = 0
LOW_TAXEL_THRESH_Y = 0
HIGH_TAXEL_THRESH_X = (NUMOFTAXELS_X - 1)
HIGH_TAXEL_THRESH_Y = (NUMOFTAXELS_Y - 1)



# import hrl_lib.util as ut
import pickle as pickle
# from hrl_lib.util import load_pickle
def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f, encoding='latin1')


class TensorPrepLib():

    def load_files_to_database(self, database_file, creation_type, verbose = False, reduce_data = False, test = False):
        # load in the training or testing files.  This may take a while.
       # print "GOT HERE!!", database_file
        dat = None
        for some_subject in database_file:
            #print creation_type, some_subject, 'some subject'
            if creation_type in some_subject:
                dat_curr = load_pickle(some_subject)
                print(some_subject, dat_curr['bed_angle_deg'][0])
                for key in dat_curr:
                    if np.array(dat_curr[key]).shape[0] != 0:
                        for inputgoalset in np.arange(len(dat_curr['images'])):
                            datcurr_to_append = dat_curr[key][inputgoalset]
                            if key == 'images' and np.shape(datcurr_to_append)[0] == 3948:
                                datcurr_to_append = list(
                                    np.array(datcurr_to_append).reshape(84, 47)[10:74, 10:37].reshape(1728))
                            try:
                                if test == False:
                                    if reduce_data == True:
                                        if inputgoalset < len(dat_curr['images'])/4:
                                            dat[key].append(datcurr_to_append)
                                    else:
                                        dat[key].append(datcurr_to_append)
                                else:
                                    if len(dat_curr['images']) == 3000:
                                        if inputgoalset < len(dat_curr['images'])/2:
                                            dat[key].append(datcurr_to_append)
                                    elif len(dat_curr['images']) == 1500:
                                        if inputgoalset < len(dat_curr['images'])/3:
                                            dat[key].append(datcurr_to_append)
                                    else:
                                        dat[key].append(datcurr_to_append)

                            except:
                                try:
                                    dat[key] = []
                                    dat[key].append(datcurr_to_append)
                                except:
                                    dat = {}
                                    dat[key] = []
                                    dat[key].append(datcurr_to_append)
            else:
                pass

        if dat is not None and verbose == True:
            for key in dat:
                print('all data keys and shape', key, np.array(dat[key]).shape)
        return dat

    def prep_images(self, im_list, dat_f, dat_m, num_repeats):
        for dat in [dat_f, dat_m]:
            if dat is not None:
                for entry in range(len(dat['images'])):
                    for i in range(num_repeats):
                        im_list.append(dat['images'][entry])
        return im_list

    def prep_reconstruction_gt(self, mesh_reconstruction_maps, dat_f, dat_m, num_repeats):
        for dat in [dat_f, dat_m]:
            if dat is not None:
                for entry in range(len(dat['images'])):
                    for i in range(num_repeats):
                        mesh_reconstruction_maps.append([dat['mesh_depth'][entry], dat['mesh_contact'][entry]*100, ])
        return np.array(mesh_reconstruction_maps)#.astype(np.int16)

    def prep_depth_input_images(self, depth_images, dat_f, dat_m, num_repeats, depth_type = 'all_meshes'):
        for dat in [dat_f, dat_m]:
            if dat is not None:
                for entry in range(len(dat['images'])):
                    for i in range(num_repeats):
                        if depth_type == 'all_meshes':
                            depth_images.append([dat['overhead_depthcam'][entry]])
                        elif depth_type == 'no_blanket':
                            depth_images.append([dat['overhead_depthcam_noblanket_noisey'][entry]])
                        elif depth_type == 'human_only':
                            depth_images.append([dat['overhead_depthcam_onlyhuman_noisey'][entry]])
                        elif depth_type == 'blanket_only':
                            depth_images.append([dat['overhead_depthcam_onlyblanket'][entry]])

        print("depth array shape: ", np.shape(depth_images), np.max(depth_images))
        return np.array(depth_images).astype(np.int16)

    def prep_reconstruction_input_est(self, reconstruction_input_est_list, dat_f, dat_m, num_repeats):
        for dat in [dat_f, dat_m]:
            if dat is not None:
                for entry in range(len(dat['images'])):
                    for i in range(num_repeats):
                        mdm_est_neg = np.copy(dat['mdm_est'][entry])
                        mdm_est_neg[mdm_est_neg > 0] = 0
                        mdm_est_neg *= -1
                        reconstruction_input_est_list.append([mdm_est_neg, dat['cm_est'][entry]*100, ])

        return np.array(reconstruction_input_est_list)#.astype(np.int16)

    def append_trainxa_besides_pmat_edges(self, train_xa, CTRL_PNL, mesh_reconstruction_maps_input_est = None, mesh_reconstruction_maps = None, depth_images = None, depth_images_out_unet = None):
        train_xa[train_xa > 0] += 1.
        train_xa = train_xa#.astype(np.int16)

        if CTRL_PNL['incl_pmat_cntct_input'] == True:
            train_contact = np.copy(train_xa[:, 0:1, :, :]) #get the pmat contact map
            train_contact[train_contact > 0] = 100

        print(np.shape(train_xa), 'shape before appending pmat contact input')
        if CTRL_PNL['recon_map_input_est'] == True:
            train_xa = np.concatenate((mesh_reconstruction_maps_input_est, train_xa), axis = 1)

        print(np.shape(train_xa), 'shape before appending pmat contact input')
        if CTRL_PNL['incl_pmat_cntct_input'] == True:
            train_xa = np.concatenate((train_contact, train_xa), axis=1)

        #for i in range(2):
        #    print np.mean(train_xa[:, i, :, :]), np.max(train_xa[:, i, :, :])


        print(np.shape(train_xa), 'shape before appending recon gt maps')
        if CTRL_PNL['recon_map_labels'] == True:
            mesh_reconstruction_maps = np.array(mesh_reconstruction_maps) #GROUND TRUTH
            train_xa = np.concatenate((train_xa, mesh_reconstruction_maps), axis=1)

        print(np.shape(train_xa), 'shape before appending input depth images. we split a single into 4')
        if CTRL_PNL['depth_in'] == True:
            #depth_images[depth_images > 0] -= 1000
            #depth_images /= 10

            print(np.max(depth_images), 'max depth ims')

            train_xa = np.concatenate((train_xa, depth_images[:, :, 0:64, 0:27]), axis = 1)
            train_xa = np.concatenate((train_xa, depth_images[:, :, 0:64, 27:54]), axis = 1)
            train_xa = np.concatenate((train_xa, depth_images[:, :, 64:128, 0:27]), axis = 1)
            train_xa = np.concatenate((train_xa, depth_images[:, :, 64:128, 27:54]), axis = 1)

        print(np.shape(train_xa), 'shape before appending input depth output unet with no blanket. we split a single into 4')
        if CTRL_PNL['depth_out_unet'] == True:
            #depth_images_out_unet[depth_images_out_unet > 0] -= 1000
            #depth_images /= 10

            print(np.max(depth_images_out_unet), 'max depth ims')#, np.std(depth_images_out_unet)

            train_xa = np.concatenate((train_xa, depth_images_out_unet[:, :, 0:64, 0:27]), axis = 1)
            train_xa = np.concatenate((train_xa, depth_images_out_unet[:, :, 0:64, 27:54]), axis = 1)
            train_xa = np.concatenate((train_xa, depth_images_out_unet[:, :, 64:128, 0:27]), axis = 1)
            train_xa = np.concatenate((train_xa, depth_images_out_unet[:, :, 64:128, 27:54]), axis = 1)

        #print np.shape(depth_images)

        #for i in range(8):
        #    print np.mean(train_xa[:, i, :, :]), np.max(train_xa[:, i, :, :])

        #print depth_images.dtype
        #print train_xa.dtype

        print("TRAIN XA SHAPE", np.shape(train_xa))
        return train_xa





    def prep_labels(self, y_flat, dat, num_repeats, z_adj, gender, is_synth, loss_vector_type, initial_angle_est, full_body_rot = False):
        if gender == "f":
            g1 = 1
            g2 = 0
        elif gender == "m":
            g1 = 0
            g2 = 1
        if is_synth == True:
            s1 = 1
        else:
            s1 = 0
        z_adj_all = np.array(24 * [0.0, 0.0, z_adj*1000])
        z_adj_one = np.array(1 * [0.0, 0.0, z_adj*1000])

        if is_synth == True and loss_vector_type != 'direct':
            if dat is not None:
                for entry in range(len(dat['markers_xyz_m'])):
                    c = np.concatenate((dat['markers_xyz_m'][entry][0:72] * 1000 + z_adj_all,
                                        dat['body_shape'][entry][0:10],
                                        dat['joint_angles'][entry][0:72],
                                        dat['root_xyz_shift'][entry][0:3] + np.array([0.0, 0.0, z_adj]),
                                        [g1], [g2], [s1],
                                        [dat['body_mass'][entry]],
                                        [(dat['body_height'][entry]-1.)*100],), axis=0)  # [x1], [x2], [x3]: female synth: 1, 0, 1.
                    if initial_angle_est == True:
                        c = np.concatenate((c,
                                            dat['betas_est'][entry][0:10],
                                            dat['angles_est'][entry][0:72],
                                            dat['root_xyz_est'][entry][0:3]), axis = 0)
                        if full_body_rot == True:
                            c = np.concatenate((c, dat['root_atan2_est'][entry][0:6]), axis = 0)
                    for i in range(num_repeats):
                        y_flat.append(c)

        elif is_synth == True and loss_vector_type == 'direct':
            if dat is not None:
                for entry in range(len(dat['markers_xyz_m_offset'])):
                    c = np.concatenate((np.array(9 * [0]),
                                        dat['markers_xyz_m_offset'][entry][3:6] * 1000 + z_adj_one,  # TORSO
                                        # fixed_torso_markers,  # TORSO
                                        dat['markers_xyz_m_offset'][entry][21:24] * 1000 + z_adj_one,  # L KNEE
                                        dat['markers_xyz_m_offset'][entry][18:21] * 1000 + z_adj_one,  # R KNEE
                                        np.array(3 * [0]),
                                        dat['markers_xyz_m_offset'][entry][27:30] * 1000 + z_adj_one,  # L ANKLE
                                        dat['markers_xyz_m_offset'][entry][24:27] * 1000 + z_adj_one,  # R ANKLE
                                        np.array(18 * [0]),
                                        dat['markers_xyz_m_offset'][entry][0:3] * 1000 + z_adj_one,  # HEAD
                                        # fixed_head_markers,
                                        np.array(6 * [0]),
                                        dat['markers_xyz_m_offset'][entry][9:12] * 1000 + z_adj_one,  # L ELBOW
                                        dat['markers_xyz_m_offset'][entry][6:9] * 1000 + z_adj_one,  # R ELBOW
                                        dat['markers_xyz_m_offset'][entry][15:18] * 1000 + z_adj_one,  # L WRIST
                                        dat['markers_xyz_m_offset'][entry][12:15] * 1000 + z_adj_one,  # R WRIST
                                        np.array(6 * [0]),
                                        np.array(85 * [0]),
                                        [g1], [g2], [s1],
                                        [dat['body_mass'][entry]],
                                        [(dat['body_height'][entry]-1.)*100],), axis=0)  # [x1], [x2], [x3]: female synth: 1, 0, 1.
                    if initial_angle_est == True:
                        c = np.concatenate((c,
                                            dat['betas_est'][entry][0:10],
                                            dat['angles_est'][entry][0:72],
                                            dat['root_xyz_est'][entry][0:3]), axis = 0)
                    for i in range(num_repeats):
                        y_flat.append(c)


        elif is_synth == False:
            if dat is not None:
                for entry in range(len(dat['markers_xyz_m'])):
                    c = np.concatenate((np.array(9 * [0]),
                                        dat['markers_xyz_m'][entry][3:6] * 1000,  # TORSO
                                        # fixed_torso_markers,  # TORSO
                                        dat['markers_xyz_m'][entry][21:24] * 1000,  # L KNEE
                                        dat['markers_xyz_m'][entry][18:21] * 1000,  # R KNEE
                                        np.array(3 * [0]),
                                        dat['markers_xyz_m'][entry][27:30] * 1000,  # L ANKLE
                                        dat['markers_xyz_m'][entry][24:27] * 1000,  # R ANKLE
                                        np.array(18 * [0]),
                                        dat['markers_xyz_m'][entry][0:3] * 1000,  # HEAD
                                        # fixed_head_markers,
                                        np.array(6 * [0]),
                                        dat['markers_xyz_m'][entry][9:12] * 1000,  # L ELBOW
                                        dat['markers_xyz_m'][entry][6:9] * 1000,  # R ELBOW
                                        dat['markers_xyz_m'][entry][15:18] * 1000,  # L WRIST
                                        dat['markers_xyz_m'][entry][12:15] * 1000,  # R WRIST
                                        np.array(6 * [0]),
                                        np.array(85 * [0]),
                                        [g1], [g2], [s1],
                                        [dat['body_mass'][entry]],
                                        [(dat['body_height'][entry]-1.)*100],), axis=0)  # [x1], [x2], [x3]: female real: 1, 0, 0.
                    if initial_angle_est == True:
                        c = np.concatenate((c,
                                            dat['betas_est'][entry][0:10],
                                            dat['angles_est'][entry][0:72],
                                            dat['root_xyz_est'][entry][0:3]), axis = 0)
                    for i in range(num_repeats):
                        y_flat.append(c)

        elif is_synth == 'real_nolabels':
            s1 = 1
            WEIGHT_LBS = 190.
            HEIGHT_IN = 73.
            weight_input = WEIGHT_LBS/2.20462
            height_input = (HEIGHT_IN*0.0254 - 1)*100

            if dat is not None:
                for entry in range(len(dat['images'])):
                    c = np.concatenate((np.array(157 * [0]),
                                        [g1], [g2], [s1],
                                        [weight_input],
                                        [height_input],), axis=0)  # [x1], [x2], [x3]: female synth: 1, 0, 1.
                    if initial_angle_est == True:
                        c = np.concatenate((c,
                                            dat['betas_est'][entry][0:10],
                                            dat['angles_est'][entry][0:72],
                                            dat['root_xyz_est'][entry][0:3]), axis = 0)
                        if full_body_rot == True:
                            c = np.concatenate((c, dat['root_atan2_est'][entry][0:6]), axis = 0)
                    for i in range(num_repeats):
                        y_flat.append(c)


        return y_flat


    def normalize_network_input(self, x, CTRL_PNL):

        for i in range(8):
            print(np.mean(x[:, i, :, :]), np.max(x[:, i, :, :]))


        if CTRL_PNL['recon_map_input_est'] == True:
            normalizing_std_constants = CTRL_PNL['norm_std_coeffs']

            if CTRL_PNL['cal_noise'] == True: normalizing_std_constants = normalizing_std_constants[1:] #here we don't precompute the contact

            for i in range(x.shape[1]):
                x[:, i, :, :] *= normalizing_std_constants[i]

        else:
            normalizing_std_constants = []
            normalizing_std_constants.append(CTRL_PNL['norm_std_coeffs'][0])
            normalizing_std_constants.append(CTRL_PNL['norm_std_coeffs'][3])
            normalizing_std_constants.append(CTRL_PNL['norm_std_coeffs'][4])
            normalizing_std_constants.append(CTRL_PNL['norm_std_coeffs'][5])
            normalizing_std_constants.append(CTRL_PNL['norm_std_coeffs'][6])


            if CTRL_PNL['cal_noise'] == True: normalizing_std_constants = normalizing_std_constants[1:] #here we don't precompute the contact

            for i in range(x.shape[1]):
                print("normalizing idx", i)
                x[:, i, :, :] *= normalizing_std_constants[i]

        for i in range(8):
            print(np.mean(x[:, i, :, :]), np.max(x[:, i, :, :]))
        return x

    def normalize_wt_ht(self, y, CTRL_PNL):
        #normalizing_std_constants = [1./30.216647403349857,
        #                             1./14.629298141231091]

        y = np.array(y)

        #y[:, 160] *= normalizing_std_constants[0]
        #y[:, 161] *= normalizing_std_constants[1]
        y[:, 160] *= CTRL_PNL['norm_std_coeffs'][7]
        y[:, 161] *= CTRL_PNL['norm_std_coeffs'][8]



        return y