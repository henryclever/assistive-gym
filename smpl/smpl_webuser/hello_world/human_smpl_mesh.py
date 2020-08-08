'''
Copyright 2015 Matthew Loper, Naureen Mahmood and the Max Planck Gesellschaft.  All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the SMPL Model license here http://smpl.is.tue.mpg.de/license

More information about SMPL is available here http://smpl.is.tue.mpg.
For comments or questions, please email us at: smpl@tuebingen.mpg.de


Please Note:
============
This is a demo version of the script for driving the SMPL model with python.
We would be happy to receive comments, help and suggestions on improving this code 
and in making it available on more platforms. 


System Requirements:
====================
Operating system: OSX, Linux

Python Dependencies:
- Numpy & Scipy  [http://www.scipy.org/scipylib/download.html]
- Chumpy [https://github.com/mattloper/chumpy]


About the Script:
=================
This script demonstrates a few basic functions to help users get started with using 
the SMPL model. The code shows how to:
  - Load the SMPL model
  - Edit pose & shape parameters of the model to create a new body in a new pose
  - Save the resulting body as a mesh in .OBJ format


Running the Hello World code:
=============================
Inside Terminal, navigate to the smpl/webuser/hello_world directory. You can run 
the hello world script now by typing the following:
>	python hello_smpl.py

'''

from smpl.smpl_webuser.serialization import load_model
import numpy as np

## Load SMPL model (here we load the female model)
## Make sure path is correct
m = load_model( '../../models/basicModel_m_lbs_10_207_0_v1.0.0.pkl' )


m.pose[0] = np.pi #pitch rotation of the person in space. 0 means the person is upside down facing back. pi is standing up facing forward
m.pose[1] = 0 #roll of the person in space. -pi/2 means they are tilted to their right side
m.pose[2] = 0#np.pi/2 #-np.pi/4 #yaw of the person in space, like turning around normal to the ground

m.pose[3] = 0 #left hip extension (i.e. leg bends back for np.pi/2)
m.pose[4] = 0 #left leg yaw about hip, where np.pi/2 makes bowed leg
m.pose[5] = 0 #left leg abduction /adduction

m.pose[6] = 0 #right hip extension (i.e. leg bends back for np.pi/2)

m.pose[9] = 0 #bending of spine at hips. np.pi/2 means person bends down to touch the ground
m.pose[10] = 0 #twisting of spine at hips. body above spine yaws normal to the ground
m.pose[11] = 0 #bending of spine at hips. np.pi/2 means person bends down sideways to touch the ground

m.pose[12] = 0 #left knee extension. (i.e. knee bends back for np.pi/2)
m.pose[13] = 0 #twisting of knee normal to ground. KEEP AT ZERO
m.pose[14] = 0 #bending of knee sideways. KEEP AT ZERO

m.pose[15] = 0 #right knee extension (i.e. knee bends back for np.pi/2)

m.pose[18] = 0 #bending at mid spine. makes person into a hunchback for positive values
m.pose[19] = 0#twisting of midspine. body above midspine yaws normal to the ground
m.pose[20] = 0 #bending of midspine, np.pi/2 means person bends down sideways to touch ground

m.pose[21] = 0 #left ankle flexion/extension
m.pose[22] = 0 #left ankle yaw about leg
m.pose[23] = 0 #left ankle twist KEEP CLOSE TO ZERO

m.pose[24] = 0 #right ankle flexion/extension
m.pose[25] = 0 #right ankle yaw about leg
m.pose[26] = 0 #right ankle twist KEEP CLOSE TO ZERO

m.pose[27] = 0 #bending at upperspine. makes person into a hunchback for positive values
m.pose[28] = 0#twisting of upperspine. body above upperspine yaws normal to the ground
m.pose[29] = 0 #bending of upperspine, np.pi/2 means person bends down sideways to touch ground

m.pose[30] = 0 #flexion/extension of left ankle midpoint

m.pose[33] = 0 #flexion/extension of right ankle midpoint

m.pose[36] = 0 #flexion/extension of lower neck. i.e. whiplash
m.pose[37] = 0 #yaw of neck

m.pose[39] = 0 #left inner shoulder roll
m.pose[40] = 0 #left inner shoulder yaw, negative moves forward
m.pose[41] = -np.pi/6 #left inner shoulder pitch, positive moves up

m.pose[42] = 0
m.pose[43] = 0 #right inner shoulder yaw, positive moves forward
m.pose[44] = 0 #right inner shoulder pitch, positive moves down

m.pose[45] = 0 #flexion/extension of upper neck


##########################ZACKORY EDIT THE NEXT FEW LINES TO CHANGE THE ARM POSE############################

m.pose[48] = 0 #left outer shoulder roll
m.pose[50] = -np.pi/3 #left outer shoulder pitch

m.pose[51] = -.3 #right outer shoulder roll
m.pose[52] = .8
m.pose[53] = .6

m.pose[54] = 0 #left elbow roll KEEP AT ZERO
m.pose[55] = 0 #left elbow flexion/extension. KEEP NEGATIVE
m.pose[56] = 0 #left elbow KEEP AT ZERO

m.pose[57] = np.pi/3
m.pose[58] = np.pi/6 #right elbow flexsion/extension KEEP POSITIVE

m.pose[60] = 0 #left wrist roll

m.pose[63] = 0 #right wrist roll
m.pose[65] = np.pi/5

m.pose[66] = 0 #left hand roll

m.pose[69] = 0 #right hand roll
m.pose[71] = np.pi/5 #right fist


m.betas[0] = 2. #overall body size. more positive number makes smaller, negative makes larger with bigger belly
m.betas[1] = 0. #positive number makes person very skinny, negative makes fat
m.betas[2] = 3. #muscle mass. higher makes person less physically fit
m.betas[3] = 0. #proportion for upper vs lower bone lengths. more negative number makes legs much bigger than arms
m.betas[4] = 3. #neck. more negative seems to make neck longer and body more skinny
m.betas[5] = 1. #size of hips. larger means bigger hips
m.betas[6] = 0. #proportion of belly with respect to rest of the body. higher number is larger belly
m.betas[8] = -4.
m.betas[9] = 0


import lib_render as libRender

libRender.standard_render(m)


#vertices
vertices = np.array(m.r)

#faces
faces = np.array(m.f) #or maybe its m.faces; I can't remember

## Write to an .obj file
outmesh_path = './hello_smpl.obj'
with open( outmesh_path, 'w') as fp:
    for v in m.r:
        fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )

    for f in m.f+1: # Faces are 1-based, not 0-based in obj files
        fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )

## Print message
print '..Output mesh saved to: ', outmesh_path 
