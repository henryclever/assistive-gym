from .bed_bathing import BedBathingEnv
from .agents import pr2, baxter, sawyer, jaco, human
from .agents.pr2 import PR2
from .agents.baxter import Baxter
from .agents.sawyer import Sawyer
from .agents.jaco import Jaco
from .agents.human import Human, HumanMesh

robot_arm = 'left'
human_controllable_joint_indices = human.right_arm_joints
class BedBathingPR2Env(BedBathingEnv):
    def __init__(self):
        super(BedBathingPR2Env, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class BedBathingMeshPR2Env(BedBathingEnv):
    def __init__(self):
        dataset_info_dict = {}
        dataset_info_dict['gender'] = 'female'
        dataset_info_dict['data_ct'] = [1501, 2000, 2109]
        dataset_info_dict['posture'] = 'general_supine'
        dataset_info_dict['set_num'] = 14
        super(BedBathingMeshPR2Env, self).__init__(robot=PR2(robot_arm), human=HumanMesh(dataset_info_dict), bed_type='pressuresim')
        #super(BedBathingMeshPR2Env, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=False), bed_type='pressuresim')

class BedBathingBaxterEnv(BedBathingEnv):
    def __init__(self):
        super(BedBathingBaxterEnv, self).__init__(robot=Baxter(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class BedBathingSawyerEnv(BedBathingEnv):
    def __init__(self):
        super(BedBathingSawyerEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class BedBathingJacoEnv(BedBathingEnv):
    def __init__(self):
        super(BedBathingJacoEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class BedBathingPR2HumanEnv(BedBathingEnv):
    def __init__(self):
        super(BedBathingPR2HumanEnv, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))

class BedBathingBaxterHumanEnv(BedBathingEnv):
    def __init__(self):
        super(BedBathingBaxterHumanEnv, self).__init__(robot=Baxter(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))

class BedBathingSawyerHumanEnv(BedBathingEnv):
    def __init__(self):
        super(BedBathingSawyerHumanEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))

class BedBathingJacoHumanEnv(BedBathingEnv):
    def __init__(self):
        super(BedBathingJacoHumanEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))

