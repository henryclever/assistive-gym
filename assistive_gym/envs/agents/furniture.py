import os
import pybullet as p
import numpy as np
from .agent import Agent

class Furniture(Agent):
    def __init__(self):
        super(Furniture, self).__init__()

    def init(self, furniture_type, directory, id, np_random, wheelchair_mounted=False):
        print(directory, "DIRECTORY")
        if 'wheelchair' in furniture_type:
            furniture = p.loadURDF(os.path.join(directory, 'wheelchair', 'wheelchair.urdf' if not wheelchair_mounted else 'wheelchair_jaco.urdf'), basePosition=[0, 0, 0.06], baseOrientation=p.getQuaternionFromEuler([np.pi/2.0, 0, np.pi], physicsClientId=id), physicsClientId=id)
        elif furniture_type == 'bed':
            furniture = p.loadURDF(os.path.join(directory, 'bed', 'bed.urdf'), basePosition=[-0.1, 0, 0], baseOrientation=p.getQuaternionFromEuler([np.pi/2.0, 0, 0], physicsClientId=id), physicsClientId=id)

            # mesh_scale = [1.1]*3
            # bed_visual = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=os.path.join(self.directory, 'bed', 'bed_single_reduced.obj'), rgbaColor=[1, 1, 1, 1], specularColor=[0.2, 0.2, 0.2], meshScale=mesh_scale, physicsClientId=self.id)
            # bed_collision = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=os.path.join(self.directory, 'bed', 'bed_single_reduced_vhacd.obj'), meshScale=mesh_scale, flags=p.GEOM_FORCE_CONCAVE_TRIMESH, physicsClientId=self.id)
            # furniture = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=bed_collision, baseVisualShapeIndex=bed_visual, basePosition=[0, 0, 0], useMaximalCoordinates=True, physicsClientId=self.id)
            # # Initialize bed position
            # p.resetBasePositionAndOrientation(furniture, [-0.1, 0, 0], p.getQuaternionFromEuler([np.pi/2.0, 0, 0], physicsClientId=self.id), physicsClientId=self.id)
        elif furniture_type == 'bed_mesh':
            pmat_start_x_border = -0.0254*3
            pmat_start_y_border = -0.0254*2.5
            shift_x = pmat_start_x_border - 0.45775
            shift_y = pmat_start_y_border - 0.98504

            #furniture = p.loadURDF(os.path.join(directory, 'bed_mesh', 'bed_mesh.urdf'), basePosition=[shift_x, shift_y, 0.3048+0.075], baseOrientation=p.getQuaternionFromEuler([0.0, 0, 0], physicsClientId=id), physicsClientId=id)
            furniture = p.loadURDF(directory+'bed_mesh.urdf', basePosition=[shift_x, shift_y, 0.3048+0.075], baseOrientation=p.getQuaternionFromEuler([0.0, 0, 0], physicsClientId=id), physicsClientId=id)


        elif furniture_type == 'table':
            furniture = p.loadURDF(os.path.join(directory, 'table', 'table_tall.urdf'), basePosition=[0.25, -0.9, 0], baseOrientation=[0, 0, 0, 1], physicsClientId=id)
        elif furniture_type == 'bowl':
            bowl_pos = np.array([-0.15, -0.55, 0.75]) + np.array([np_random.uniform(-0.05, 0.05), np_random.uniform(-0.05, 0.05), 0])
            furniture = p.loadURDF(os.path.join(directory, 'dinnerware', 'bowl.urdf'), basePosition=bowl_pos, baseOrientation=p.getQuaternionFromEuler([np.pi/2.0, 0, 0], physicsClientId=id), physicsClientId=id)
        elif furniture_type == 'nightstand':
            furniture = p.loadURDF(os.path.join(directory, 'nightstand', 'nightstand.urdf'), basePosition=np.array([-0.9, 0.7, 0]), baseOrientation=p.getQuaternionFromEuler([np.pi/2.0, 0, 0], physicsClientId=id), physicsClientId=id)
        else:
            furniture = None

        super(Furniture, self).init(furniture, id, np_random, indices=-1)

