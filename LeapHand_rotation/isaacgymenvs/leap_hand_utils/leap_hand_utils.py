'''
Some utilities for LEAP Hand that help with converting joint angles between each convention.
'''
from re import L
import numpy as np


'''
Embodiments:

LEAPhand: Real LEAP hand (180 for the motor is actual zero)
LEAPsim:  Leap hand in sim (has allegro-like zero positions)
one_range: [-1, 1] for all joints to facilitate RL
allegro:  Allegro hand in real or sim
'''

#Safety clips all joints so nothing unsafe can happen. Highly recommend using this before commanding
def angle_safety_clip(joints):
    sim_min, sim_max = LEAPsim_limits()
    real_min = LEAPsim_to_LEAPhand(sim_min)
    real_max = LEAPsim_to_LEAPhand(sim_max)
    return np.clip(joints, real_min, real_max)

###Sometimes it's useful to constrain the thumb more heavily(you have to implement here), but regular usually works good.
def LEAPsim_limits(type = "regular"):
    sim_min = np.array([-1.0470, -0.3140, -0.5060, -0.3660, -1.0470, -0.3140, -0.5060, -0.3660,
         -1.0470, -0.3140, -0.5060, -0.3660, -0.3490, -0.4700, -1.2000, -1.3400])
    sim_max = np.array([1.0470, 2.2300, 1.8850, 2.0420, 1.0470, 2.2300, 1.8850, 2.0420, 1.0470,
         2.2300, 1.8850, 2.0420, 2.0940, 2.4430, 1.9000, 1.8800])
    return sim_min, sim_max

#this goes from [-1, 1] to [lower, upper]
def scale(x, lower, upper):
    return (0.5 * (x + 1.0) * (upper - lower) + lower)
#this goes from [lower, upper] to [-1, 1]
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower)/(upper - lower)

#-----------------------------------------------------------------------------------
#Isaac has custom ranges from -1 to 1 so we convert that to LEAPHand real world
def sim_ones_to_LEAPhand(joints, hack_thumb = False):
    sim_min, sim_max = LEAPsim_limits(type = hack_thumb)
    joints = scale(joints, sim_min, sim_max)
    joints = LEAPsim_to_LEAPhand(joints)
    return joints
#LEAPHand real world to Isaac has custom ranges from -1 to 1
def LEAPhand_to_sim_ones(joints, hack_thumb = False):  
    joints = LEAPhand_to_LEAPsim(joints)
    sim_min, sim_max = LEAPsim_limits(type = hack_thumb)
    joints = unscale(joints, sim_min, sim_max)
    return joints

#-----------------------------------------------------------------------------------
###Sim LEAP hand to real leap hand  Sim is allegro-like but all 16 joints are usable.
def LEAPsim_to_LEAPhand(joints):
    joints = np.array(joints)
    ret_joints = joints + 3.14159
    return ret_joints
###Real LEAP hand to sim leap hand  Sim is allegro-like but all 16 joints are usable.
def LEAPhand_to_LEAPsim(joints):
    joints = np.array(joints)
    ret_joints = joints - 3.14159
    return ret_joints

#-----------------------------------------------------------------------------------
#Converts allegrohand radians to LEAP (radians)
#Only converts the joints that match, all 4 of the thumb and the outer 3 for each of the other fingers
#All the clockwise/counterclockwise signs are the same between the two hands.  Just the offset (mostly 180 degrees off)
def allegro_to_LEAPhand(joints, teleop = False, zeros = True):
    joints = np.array(joints)
    ret_joints = joints + 3.14159
    if zeros:
        ret_joints[0] = ret_joints[4] = ret_joints[8] = 3.14
    if teleop:
        ret_joints[12] = joints[12] + 0.2 
        ret_joints[14] = joints[14] - 0.2   
    return ret_joints
# Converts LEAP to allegrohand (radians)
def LEAPhand_to_allegro(joints, teleop = False, zeros = True):
    joints = np.array(joints)
    ret_joints = joints - 3.14159
    if zeros:
        ret_joints[0] = ret_joints[4] = ret_joints[8] = 0
    if teleop:
        ret_joints[12] = joints[12] - 0.2
        ret_joints[14] = joints[14] + 0.2    
    return ret_joints
#-----------------------------------------------------------------------------------
