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

from scipy.ndimage.filters import gaussian_filter


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
NUMOFTAXELS_X = 64#73 #taxels
NUMOFTAXELS_Y = 27#30
INTER_SENSOR_DISTANCE = 0.0286#metres
LOW_TAXEL_THRESH_X = 0
LOW_TAXEL_THRESH_Y = 0
HIGH_TAXEL_THRESH_X = (NUMOFTAXELS_X - 1)
HIGH_TAXEL_THRESH_Y = (NUMOFTAXELS_Y - 1)



class PreprocessingLib():
    def __init__(self):

        self.camera_to_bed_dist = 1.645 - 0.2032
        #zero_location += 0.5
        #zero_location = zero_location.astype(int)

        self.x = np.arange(0, 54).astype(float)
        self.x = np.tile(self.x, (128, 1))
        self.y = np.arange(0, 128).astype(float)
        self.y = np.tile(self.y, (54, 1)).T

        self.x_coord_from_camcenter = self.x - 26.5  # self.depthcam_midpixel[0]
        self.y_coord_from_camcenter = self.y - 63.5  # self.depthcam_midpixel[1]


    def eulerAnglesToRotationMatrix(self,theta):
        R_x = np.array([[1, 0, 0],
                        [0, math.cos(theta[0]), -math.sin(theta[0])],
                        [0, math.sin(theta[0]), math.cos(theta[0])]
                        ])

        R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                        [0, 1, 0],
                        [-math.sin(theta[1]), 0, math.cos(theta[1])]
                        ])

        R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                        [math.sin(theta[2]), math.cos(theta[2]), 0],
                        [0, 0, 1]
                        ])

        R = np.dot(R_z, np.dot(R_y, R_x))

        return R

    def preprocessing_add_image_noise(self, images, pmat_chan_idx, norm_std_coeffs):

        queue = np.copy(images[:, pmat_chan_idx:pmat_chan_idx+2, :, :])
        queue[queue != 0] = 1.


        x = np.arange(-10, 10)
        xU, xL = x + 0.5, x - 0.5
        prob = ss.norm.cdf(xU, scale=1) - ss.norm.cdf(xL,scale=1)  # scale is the standard deviation using a cumulative density function
        prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
        image_noise = np.random.choice(x, size=(images.shape[0], 2, images.shape[2], images.shape[3]), p=prob)

        
        image_noise = image_noise*queue
        image_noise = image_noise.astype(float)
        #image_noise[:, 0, :, :] /= 11.70153502792190
        #image_noise[:, 1, :, :] /= 45.61635847182483
        image_noise[:, 0, :, :] *= norm_std_coeffs[3]
        image_noise[:, 1, :, :] *= norm_std_coeffs[4]

        images[:, pmat_chan_idx:pmat_chan_idx+2, :, :] += image_noise

        #print images[0, 0, 50, 10:25], 'added noise'

        #clip noise so we dont go outside sensor limits
        #images[:, pmat_chan_idx, :, :] = np.clip(images[:, pmat_chan_idx, :, :], 0, 100/11.70153502792190)
        #images[:, pmat_chan_idx, :, :] = np.clip(images[:, pmat_chan_idx, :, :], 0, 100.*norm_std_coeffs[4])
        images[:, pmat_chan_idx, :, :] = np.clip(images[:, pmat_chan_idx, :, :], 0, 10000.)
        #images[:, pmat_chan_idx+1, :, :] = np.clip(images[:, pmat_chan_idx+1, :, :], 0, 10000)
        return images


    def preprocessing_add_calibration_noise(self, images, pmat_chan_idx, norm_std_coeffs, is_training, noise_amount, normalize_per_image):
        time_orig = time.time()
        if is_training == True:
            variation_amount = float(noise_amount)
            print("ADDING CALIB NOISE", variation_amount)

            #pmat_contact_orig = np.copy(images[:, pmat_chan_idx, :, :])
            #pmat_contact_orig[pmat_contact_orig != 0] = 1.
            #sobel_contact_orig = np.copy(images[:, pmat_chan_idx+1, :, :])
            #sobel_contact_orig[sobel_contact_orig != 0] = 1.

            for map_index in range(images.shape[0]):

                pmat_contact_orig = np.copy(images[map_index, pmat_chan_idx, :, :])
                pmat_contact_orig[pmat_contact_orig != 0] = 1.
                sobel_contact_orig = np.copy(images[map_index, pmat_chan_idx + 1, :, :])
                sobel_contact_orig[sobel_contact_orig != 0] = 1.


                # first multiply
                amount_to_mult_im = random.normalvariate(mu = 1.0, sigma = variation_amount) #mult a variation of 10%
                amount_to_mult_sobel = random.normalvariate(mu = 1.0, sigma = variation_amount) #mult a variation of 10%
                images[map_index, pmat_chan_idx, :, :] = images[map_index, pmat_chan_idx, :, :] * amount_to_mult_im
                images[map_index, pmat_chan_idx+1, :, :] = images[map_index, pmat_chan_idx+1, :, :] * amount_to_mult_sobel

                # then add
                #amount_to_add_im = random.normalvariate(mu = 0.0, sigma = (1./11.70153502792190)*(98.666 - 0.0)*0.1) #add a variation of 10% of the range
                #amount_to_add_sobel = random.normalvariate(mu = 0.0, sigma = (1./45.61635847182483)*(386.509 - 0.0)*0.1) #add a variation of 10% of the range

                if normalize_per_image == True:
                    amount_to_add_im = random.normalvariate(mu = 0.0, sigma = norm_std_coeffs[3]*(70. - 0.0)*variation_amount) #add a variation of 10% of the range
                    amount_to_add_sobel = random.normalvariate(mu = 0.0, sigma = norm_std_coeffs[4]*(70. - 0.0)*variation_amount) #add a variation of 10% of the range
                else:
                    amount_to_add_im = random.normalvariate(mu = 0.0, sigma = norm_std_coeffs[3]*(98.666 - 0.0)*variation_amount) #add a variation of 10% of the range
                    amount_to_add_sobel = random.normalvariate(mu = 0.0, sigma = norm_std_coeffs[4]*(386.509 - 0.0)*variation_amount) #add a variation of 10% of the range

                images[map_index, pmat_chan_idx, :, :] = images[map_index, pmat_chan_idx, :, :] + amount_to_add_im
                images[map_index, pmat_chan_idx+1, :, :] = images[map_index, pmat_chan_idx+1, :, :] + amount_to_add_sobel
                images[map_index, pmat_chan_idx, :, :] = np.clip(images[map_index, pmat_chan_idx, :, :], a_min = 0., a_max = 10000)
                images[map_index, pmat_chan_idx+1, :, :] = np.clip(images[map_index, pmat_chan_idx+1, :, :], a_min = 0., a_max = 10000)

                #cut out the background. need to do this after adding.
                images[map_index, pmat_chan_idx, :, :] *= pmat_contact_orig#[map_index, :, :]
                images[map_index, pmat_chan_idx+1, :, :] *= sobel_contact_orig#[map_index, :, :]


                amount_to_gauss_filter_im = random.normalvariate(mu = 0.5, sigma = variation_amount)
                amount_to_gauss_filter_sobel = random.normalvariate(mu = 0.5, sigma = variation_amount)
                images[map_index, pmat_chan_idx, :, :] = gaussian_filter(images[map_index, pmat_chan_idx, :, :], sigma= amount_to_gauss_filter_im) #pmap
                images[map_index, pmat_chan_idx+1, :, :] = gaussian_filter(images[map_index, pmat_chan_idx+1, :, :], sigma= amount_to_gauss_filter_sobel) #sobel #NOW


        else:  #if its NOT training we should still blur things by 0.5
            print(pmat_chan_idx, np.shape(images), 'pmat chan idx')

            for map_index in range(images.shape[0]):
               # print pmat_chan_idx, images.shape, 'SHAPE'
                images[map_index, pmat_chan_idx, :, :] = gaussian_filter(images[map_index, pmat_chan_idx, :, :], sigma= 0.5) #pmap
                images[map_index, pmat_chan_idx+1, :, :] = gaussian_filter(images[map_index, pmat_chan_idx+1, :, :], sigma= 0.5) #sobel

        #images[:, pmat_chan_idx, :, :] = np.clip(images[:, pmat_chan_idx, :, :], 0, 100/11.70153502792190)
            if normalize_per_image == False:
                images[:, pmat_chan_idx, :, :] = np.clip(images[:, pmat_chan_idx, :, :], 0, 100.*norm_std_coeffs[3])
            else:
                images[:, pmat_chan_idx, :, :] = np.clip(images[:, pmat_chan_idx, :, :], 0, 10000.)



        #now calculate the contact map AFTER we've blurred it
        pmat_contact = np.copy(images[:, pmat_chan_idx:pmat_chan_idx+1, :, :])
        #pmat_contact[pmat_contact != 0] = 100./41.80684362163343
        pmat_contact[pmat_contact != 0] = 100.*norm_std_coeffs[0]
        images = np.concatenate((pmat_contact, images), axis = 1)

        #for i in range(0, 20):
        #    VisualizationLib().visualize_pressure_map(images[i, 0, :, :] * 20., None, None,
        #                                              images[i, 1, :, :] * 20., None, None,
        #                                              block=False)
        #    time.sleep(0.5)


        #print time.time() - time_orig

        return images

    def multiply_along_axis(self, A, B, axis):
        A = np.array(A)
        B = np.array(B)
        # shape check
        if axis >= A.ndim:
            raise AxisError(axis, A.ndim)
        if A.shape[axis] != B.size:
            raise ValueError("'A' and 'B' must have the same length along the given axis")
        # Expand the 'B' according to 'axis':
        # 1. Swap the given axis with axis=0 (just need the swapped 'shape' tuple here)
        swapped_shape = A.swapaxes(0, axis).shape
        # 2. Repeat:
        # loop through the number of A's dimensions, at each step:
        # a) repeat 'B':
        #    The number of repetition = the length of 'A' along the
        #    current looping step;
        #    The axis along which the values are repeated. This is always axis=0,
        #    because 'B' initially has just 1 dimension
        # b) reshape 'B':
        #    'B' is then reshaped as the shape of 'A'. But this 'shape' only
        #     contains the dimensions that have been counted by the loop
        for dim_step in range(A.ndim - 1):
            B = B.repeat(swapped_shape[dim_step + 1], axis=0) \
                .reshape(swapped_shape[:dim_step + 2])
        # 3. Swap the axis back to ensure the returned 'B' has exactly the
        # same shape of 'A'
        B = B.swapaxes(0, axis)
        return A * B

    def preprocessing_add_depth_calnoise(self, depth_tensor, noise_amount):

        depth_images = depth_tensor.data.numpy().astype(np.int16)
        variation_amount = float(noise_amount)

        print("ADDING DEPTH NOISE")

        depth_images[depth_images < 1000] = 0

        mask_foreground = np.copy(depth_images)
        mask_foreground[mask_foreground != 0] = 1
        mask_background = mask_foreground - 1
        mask_background[mask_background != 0] = 1


        #batch_max = np.amax(depth_images.reshape(-1, depth_images.shape[1]*depth_images.shape[2]*depth_images.shape[3]), axis = 1)
        #batch_nonzeroct = np.count_nonzero(depth_images.reshape(-1, depth_images.shape[1]*depth_images.shape[2]*depth_images.shape[3]), axis = 1)
        #batch_sum = np.sum(depth_images.reshape(-1, depth_images.shape[1]*depth_images.shape[2]*depth_images.shape[3]), axis = 1)
        #depth_images_mod = np.copy(depth_images)


        batch_fake_floor = np.random.uniform(low = 1219.5, high = 2438.5, size = depth_images.shape[0]).astype(np.int16) #vary floor between 4 and 8 feet from camera
        add_background = self.multiply_along_axis(mask_background, batch_fake_floor,  axis = 0)

        depth_images += add_background

        #print depth_images[0, :]
        #print add_background.shape
        #print add_background[0, :]
        #print batch_fake_floor[0]

        #print depth_images.shape
        #print batch_max.shape



        #white noise
        queue = np.copy(depth_images[:, 0:1, :, :])
        queue[queue != 0] = 1.
        x = np.arange(-10, 10)
        xU, xL = x + 0.5, x - 0.5
        prob = ss.norm.cdf(xU, scale=1) - ss.norm.cdf(xL,scale=1)  # scale is the standard deviation using a cumulative density function
        prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
        image_noise = np.random.choice(x, size=(depth_images.shape[0], 1, depth_images.shape[2], depth_images.shape[3]), p=prob)
        image_noise = image_noise * queue
        image_noise = image_noise.astype(float)
        image_noise *= 24.83567318518728


        print(image_noise.shape, 'im noise shape')
        batch_vary_bg_noise = np.random.uniform(low=0, high=10, size=depth_images.shape[0])
        image_noise = image_noise * (mask_foreground + self.multiply_along_axis(mask_background, batch_vary_bg_noise,  axis = 0))

        depth_images += image_noise.astype(np.int16)
        depth_images = np.clip(depth_images, 0, 10000.)


        depth_tensor = torch.Tensor(depth_images)
        return depth_tensor

    def preprocessing_blur_images(self, x_data, mat_size, sigma):

        x_data_return = []
        for map_index in range(len(x_data)):
            p_map = np.reshape(x_data[map_index], mat_size)

            p_map = gaussian_filter(p_map, sigma= sigma)

            x_data_return.append(p_map.flatten())

        return x_data_return




    def preprocessing_create_pressure_angle_stack(self,x_data, mat_size, CTRL_PNL):
        '''This is for creating a 2-channel input using the height of the bed. '''

        if CTRL_PNL['verbose']: print(np.max(x_data))
        x_data = np.clip(x_data, 0, 100)

        print("normalizing per image", CTRL_PNL['normalize_per_image'])

        p_map_dataset = []
        for map_index in range(len(x_data)):
            # print map_index, self.mat_size, 'mapidx'
            # Resize mat to make into a matrix

            p_map = np.reshape(x_data[map_index], mat_size)

            if CTRL_PNL['normalize_per_image'] == True:
                p_map = p_map * (20000./np.sum(p_map))

            if mat_size == (84, 47):
                p_map = p_map[10:74, 10:37]

            # this makes a sobel edge on the image
            sx = ndimage.sobel(p_map, axis=0, mode='constant')
            sy = ndimage.sobel(p_map, axis=1, mode='constant')
            p_map_inter = np.hypot(sx, sy)
            if CTRL_PNL['clip_sobel'] == True:
                p_map_inter = np.clip(p_map_inter, a_min=0, a_max = 100)

            if CTRL_PNL['normalize_per_image'] == True:
                p_map_inter = p_map_inter * (20000. / np.sum(p_map_inter))

            #print np.sum(p_map), 'sum after norm'
            p_map_dataset.append([p_map, p_map_inter])

        return p_map_dataset


    def preprocessing_pressure_map_upsample(self, data, multiple, order=1):
        '''Will upsample an incoming pressure map dataset'''
        p_map_highres_dataset = []


        if len(np.shape(data)) == 3:
            for map_index in range(len(data)):
                #Upsample the current map using bilinear interpolation
                p_map_highres_dataset.append(
                        ndimage.zoom(data[map_index], multiple, order=order))
        elif len(np.shape(data)) == 4:
            for map_index in range(len(data)):
                p_map_highres_dataset_subindex = []
                for map_subindex in range(len(data[map_index])):
                    #Upsample the current map using bilinear interpolation
                    p_map_highres_dataset_subindex.append(ndimage.zoom(data[map_index][map_subindex], multiple, order=order))
                p_map_highres_dataset.append(p_map_highres_dataset_subindex)

        return p_map_highres_dataset



    def pad_pressure_mats(self,NxHxWimages):
        padded = np.zeros((NxHxWimages.shape[0],NxHxWimages.shape[1]+20,NxHxWimages.shape[2]+20))
        padded[:,10:74,10:37] = NxHxWimages
        NxHxWimages = padded
        return NxHxWimages


    def preprocessing_per_im_norm(self, images, CTRL_PNL):

        if CTRL_PNL['depth_map_input_est'] == True:
            pmat_sum = 1./(torch.sum(torch.sum(images[:, 3, :, :], dim=1), dim=1)/100000.)
            sobel_sum = 1./(torch.sum(torch.sum(images[:, 4, :, :], dim=1), dim=1)/100000.)

            print("ConvNet input size: ", images.size(), pmat_sum.size())
            for i in range(images.size()[1]):
                print(i, torch.min(images[0, i, :, :]), torch.max(images[0, i, :, :]))

            images[:, 3, :, :] = (images[:, 3, :, :].permute(1, 2, 0)*pmat_sum).permute(2, 0, 1)
            images[:, 4, :, :] = (images[:, 4, :, :].permute(1, 2, 0)*sobel_sum).permute(2, 0, 1)

        else:
            pmat_sum = 1./(torch.sum(torch.sum(images[:, 1, :, :], dim=1), dim=1)/100000.)
            sobel_sum = 1./(torch.sum(torch.sum(images[:, 2, :, :], dim=1), dim=1)/100000.)

            print("ConvNet input size: ", images.size(), pmat_sum.size())
            for i in range(images.size()[1]):
                print(i, torch.min(images[0, i, :, :]), torch.max(images[0, i, :, :]))


            images[:, 1, :, :] = (images[:, 1, :, :].permute(1, 2, 0)*pmat_sum).permute(2, 0, 1)
            images[:, 2, :, :] = (images[:, 2, :, :].permute(1, 2, 0)*sobel_sum).permute(2, 0, 1)



        #do this ONLY to pressure and sobel. scale the others to get them in a reasonable range, by a constant factor.


        return images