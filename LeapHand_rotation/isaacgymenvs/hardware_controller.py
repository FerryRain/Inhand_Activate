import numpy as np
from LeapHand_rotation.isaacgymenvs.leap_node import LeapNode


def unscale_np(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)

class LeapHand(object):
    def __init__(self):

        self.hand = LeapNode()

        self.leap_dof_lower = None
        self.leap_dof_upper = None
        self.sim_to_real_indices = None
        self.real_to_sim_indices = None

    def sim_to_real(self, values):
        return values[self.sim_to_real_indices]

    def real_to_sim(self, values):
        return values[self.real_to_sim_indices]

    def command_joint_position(self, desired_pose):

        if (not hasattr(desired_pose, '__len__') or len(desired_pose) != 16):
            print(f'Desired pose must be a 16-d array: got {desired_pose}')
            return False

        scaled_pose = (2 * desired_pose - self.leap_dof_lower - self.leap_dof_upper) / (self.leap_dof_upper - self.leap_dof_lower)
        
        reordered_pose = self.sim_to_real(scaled_pose)

       
        self.hand.set_ones(reordered_pose)
        return True
        

    def poll_joint_position(self):

        joint_position_raw = np.array(self.hand.read_pos())
        joint_position_sim_scaled = self.LEAPhand_to_sim_ones(joint_position_raw)
        
        joint_position_reordered = self.real_to_sim(joint_position_sim_scaled)
        
        final_joint_position = (self.leap_dof_upper - self.leap_dof_lower) * (joint_position_reordered + 1) / 2 + self.leap_dof_lower

        return (final_joint_position, None) 

    def LEAPsim_limits(self):
        sim_min = self.sim_to_real(self.leap_dof_lower)
        sim_max = self.sim_to_real(self.leap_dof_upper)
        return sim_min, sim_max

    def LEAPhand_to_LEAPsim(self, joints):
        joints = np.array(joints)
        ret_joints = joints - 3.14
        return ret_joints

    def LEAPhand_to_sim_ones(self, joints):
        joints = self.LEAPhand_to_LEAPsim(joints)
        sim_min, sim_max = self.LEAPsim_limits()
        joints = unscale_np(joints, sim_min, sim_max)
        return joints