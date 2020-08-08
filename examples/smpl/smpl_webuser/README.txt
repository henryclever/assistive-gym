License:
========
To learn about SMPL, please visit our website: http://smpl.is.tue.mpg
You can find the SMPL paper at: http://files.is.tue.mpg.de/black/papers/SMPL2015.pdf

Visit our downloads page to download some sample animation files (FBX), and python code:
http://smpl.is.tue.mpg/downloads

For comments or questions, please email us at: smpl@tuebingen.mpg.de


System Requirements:
====================
Operating system: OSX, Linux

Python Dependencies:
- Numpy & Scipy  [http://www.scipy.org/scipylib/download.html]
- Chumpy 		 [https://github.com/mattloper/chumpy]
- OpenCV 		 [http://opencv.org/downloads.html] 


Getting Started:
================

1. Extract the Code:
--------------------
Extract the 'smpl.zip' file to your home directory (or any other location you wish)


2. Set the PYTHONPATH:
----------------------
We need to update the PYTHONPATH environment variable so that the system knows how to find the SMPL code. Add the following lines to your ~/.bash_profile file (create it if it doesn't exist; Linux users might have ~/.bashrc file instead), replacing ~/smpl with the location where you extracted the smpl.zip file:

	SMPL_LOCATION=~/smpl
	export PYTHONPATH=$PYTHONPATH:$SMPL_LOCATION


Open a new terminal window to check if the python path has been updated by typing the following:
>  echo $PYTHONPATH


3. Run the Hello World scripts:
-------------------------------
In the new Terminal window, navigate to the smpl/smpl_webuser/hello_world directory. You can run the hello world scripts now by typing the following:

> python hello_smpl.py

OR 

> python render_smpl.py



Note:
Both of these scripts will require the dependencies listed above. The scripts are provided as a sample to help you get started. 



4. Key for functions in model object:
> J: [24, 3] matrix of joint positions <chumpy.reordering.transpose object  >
> J_regressor: <24x6890 sparse matrix of type '<type 'numpy.float64'>'
	with 226 stored elements in Compressed Sparse Column format>
	> presume it is some transformer from joint positions to mesh
> J_transformed: [24, 3] matrix of joint positions <chumpy.ch_ops.add object  >
> a: [6890, 3] matrix of mesh. <chumpy.reordering.Select object  >
> add_dterm: seems the same as show_tree
> b: [1, 3] matrix of <chumpy.reordering.Reshape object  >
> betas: [10, 1] matrix <chumpy.ch.Ch object >
> kintree_table: [2, 24] matrix ordering which parts of the tree come off what (I think!)
> loop_children_do: [6890, 3] matrix like show_tree
> show_tree: <bound method add.show_tree of <chumpy.ch_ops.add object >, some R2 matrix
    > show_tree.imfunc <function chumpy.ch.show_tree>
    > show_tree.im_class chumpy.ch_ops.add
    > show_tree.im_self <chumpy.ch_ops.add object >, some R2 matrix
        > show_tree.im_self[0, :] <chumpy.reordering.Select object >, 1 of 6890 mesh edges
> size: outputs 20670. (3 * 6890)
> weights: [6890, 24] mapping from joints to




