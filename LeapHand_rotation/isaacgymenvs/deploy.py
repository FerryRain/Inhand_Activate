# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2023 Ananye Agarwal
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on:
# https://github.com/HaozhiQi/hora/blob/main/hora/algo/deploy/deploy.py
# --------------------------------------------------------

import isaacgym
import torch
import xml.etree.ElementTree as ET
import os
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.utils import to_absolute_path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed
from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, get_rlgames_env_creator
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner, _override_sigma, _restore
from rl_games.algos_torch import model_builder
from isaacgymenvs.learning import amp_continuous
from isaacgymenvs.learning import amp_models
from isaacgymenvs.learning import amp_network_builder
from isaacgym.torch_utils import *
import numpy as np
from gym import spaces
from collections import deque
import time
import serial
import threading
import cv2
from scipy.ndimage import gaussian_filter
import copy
from pynput import keyboard
# import matplotlib.pyplot as plt
# import seaborn as sns
# os.system('cls')

contact_data_norm = np.zeros((16,16))
# WINDOW_WIDTH = 400
# WINDOW_HEIGHT = 400
# # WINDOW_WIDTH = contact_data_norm.shape[1]*30
# # WINDOW_HEIGHT = contact_data_norm.shape[0]*30
# cv2.namedWindow("Contact Data_left", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Contact Data_left",WINDOW_WIDTH, WINDOW_HEIGHT)
THRESHOLD =5
NOISE_SCALE =5

latest_tactile_tensor = torch.zeros(16)
tactile_lock = threading.Lock()
raw_data_lock = threading.Lock()
vis_exit_flag = False
flag = False

def readThread(serDev):
    global contact_data_norm, flag
    data_tac = []
    num = 0
    t1 = 0
    backup = None
    flag = False
    current = None


    EXPECTED_ROWS = 16
    EXPECTED_COLS = 16

    # --- Initialization phase: collect many frames to compute median ---
    while True:
        if serDev.in_waiting > 0:
            try:
                line = serDev.readline().decode('utf-8').strip()
            except Exception:
                line = ""
            if len(line) < 10:
                if current is not None and len(current) == EXPECTED_ROWS:
                    try:
                        arr = np.asarray(current, dtype=float)
                    except Exception as e:
                        print("Warning: cannot convert current to float array:", e)
                        current = []
                        continue
                    if arr.shape == (EXPECTED_ROWS, EXPECTED_COLS):
                        backup = arr.copy()
                        if t1 != 0:
                            print("fps", 1.0 / (time.time() - t1))
                        t1 = time.time()
                        data_tac.append(backup)
                        num += 1
                        if num > 30:
                            break
                    else:
                        print("Skipped frame with unexpected shape:", arr.shape)
                current = []
                continue

            if current is not None:
                str_values = line.split()
                try:
                    int_values = [int(val) for val in str_values]
                except ValueError:
                    current.append([0] * EXPECTED_COLS)
                    continue
                if len(int_values) != EXPECTED_COLS:
                    if len(int_values) < EXPECTED_COLS:
                        int_values = int_values + [0] * (EXPECTED_COLS - len(int_values))
                    else:
                        int_values = int_values[:EXPECTED_COLS]
                current.append(int_values)
    if len(data_tac) == 0:
        raise RuntimeError("No valid frames collected for median computation.")
    try:
        data_tac_stack = np.stack(data_tac, axis=0)   # shape (N,16,16)
    except Exception as e:
        print("Error stacking frames for median:", e)
        for i, f in enumerate(data_tac):
            print(i, type(f), getattr(f, "shape", None))
        raise

    median = np.median(data_tac_stack, axis=0).astype(float)  # shape (16,16)
    flag = True
    print("Finish Initialization!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    while True:
        if serDev.in_waiting > 0:
            try:
                line = serDev.readline().decode('utf-8').strip()
            except Exception:
                line = ""
            if len(line) < 10:
                if current is not None and len(current) == EXPECTED_ROWS:
                    try:
                        backup = np.asarray(current, dtype=float)
                    except Exception as e:
                        print("Warning: cannot convert current to float array (runtime):", e)
                        backup = None
                current = []
                if backup is not None:
                    if not isinstance(backup, np.ndarray):
                        backup = np.asarray(backup, dtype=float)
                    if not isinstance(median, np.ndarray):
                        median = np.asarray(median, dtype=float)

                    if backup.shape == median.shape:
                        try:
                            contact_data = backup - median - THRESHOLD
                        except Exception as e:
                            print("Error subtracting arrays:", e)
                            contact_data = np.asarray([[float(backup[i,j]) - float(median[i,j]) - THRESHOLD
                                                        for j in range(min(backup.shape[1], median.shape[1]))]
                                                       for i in range(min(backup.shape[0], median.shape[0]))], dtype=float)
                    else:
                        min_rows = min(backup.shape[0], median.shape[0])
                        min_cols = min(backup.shape[1], median.shape[1])
                        contact_data = backup[:min_rows, :min_cols] - median[:min_rows, :min_cols] - THRESHOLD
                        if contact_data.shape != (EXPECTED_ROWS, EXPECTED_COLS):
                            padded = np.zeros((EXPECTED_ROWS, EXPECTED_COLS), dtype=float)
                            padded[:contact_data.shape[0], :contact_data.shape[1]] = contact_data
                            contact_data = padded

                    contact_data = np.clip(contact_data, 0, 100)

                    if np.max(contact_data) < THRESHOLD:
                        contact_data_norm = contact_data / NOISE_SCALE
                    else:
                        contact_data_norm = contact_data / np.max(contact_data)

                    with raw_data_lock:
                            contact_data_norm = contact_data_norm

                continue

            if current is not None:
                str_values = line.split()
                try:
                    int_values = [int(val) for val in str_values]
                except ValueError:
                    int_values = [0] * EXPECTED_COLS
                if len(int_values) != EXPECTED_COLS:
                    if len(int_values) < EXPECTED_COLS:
                        int_values = int_values + [0] * (EXPECTED_COLS - len(int_values))
                    else:
                        int_values = int_values[:EXPECTED_COLS]
                current.append(int_values)
                continue

# PORT = "left_gripper_right_finger"
PORT ='/dev/ttyUSB0'
BAUD = 2000000
# serDev = serial.Serial(PORT,2000000)
serDev = serial.Serial(PORT,BAUD)
exitThread = False
serDev.flush()
serialThread = threading.Thread(target=readThread, args=(serDev,))
serialThread.daemon = True
serialThread.start()


def apply_gaussian_blur(contact_map, sigma=0.1):
    return gaussian_filter(contact_map, sigma=sigma)

def temporal_filter(new_frame, prev_frame, alpha=0.2):
    """
    Apply temporal smoothing filter.
    'alpha' determines the blending factor.
    A higher alpha gives more weight to the current frame, while a lower alpha gives more weight to the previous frame.
    """
    return alpha * new_frame + (1 - alpha) * prev_frame

def process_tactile_thread():
    global latest_tactile_tensor, flag, contact_data_norm
    WINDOW_WIDTH = 400
    WINDOW_HEIGHT = 400
    cv2.namedWindow("Contact Data_left", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Contact Data_left", WINDOW_WIDTH, WINDOW_HEIGHT)
    
    prev_frame = np.zeros_like(contact_data_norm, dtype=float)
    
    print("Tactile processing thread started.")

    while not vis_exit_flag:
        if flag:
            try:
                with raw_data_lock:
                    current_raw_snapshot = contact_data_norm.copy()

                temp_filtered_data = temporal_filter(contact_data_norm, prev_frame)
                prev_frame = temp_filtered_data

                temp_filtered_data_scaled = (temp_filtered_data * 255).astype(np.uint8)
                
                patch_data = temp_filtered_data_scaled[8:12, 12:16]
                
                flat_data = patch_data.reshape(16, order='F')
                flat_data = flat_data[[8, 6, 1, 10, 7, 4, 2, 9, 5, 3, 0, 11, 12, 15, 13, 14]]
                
                new_tensor = torch.from_numpy(np.where(flat_data > 50, 1, 0).astype(np.float32))
                
                with tactile_lock:
                    latest_tactile_tensor = new_tensor

                colormap = cv2.applyColorMap(flat_data, cv2.COLORMAP_VIRIDIS)
                cv2.imshow("Contact Data_left", colormap)
                cv2.waitKey(1) 
                
            except Exception as e:
                print(f"Error in tactile processing thread: {e}")
                time.sleep(0.1) 
        else:
            time.sleep(0.1)
        
        time.sleep(0.002)

# Start the processing/visualization thread immediately
vis_thread = threading.Thread(target=process_tactile_thread)
vis_thread.daemon = True
vis_thread.start()

class HardwarePlayer(object):
    def __init__(self, config):
        self.config = omegaconf_to_dict(config)
        #self.set_defaults()
        self.action_scale = 1 / 16
        
        self.num_hand_dofs = 16
        self.num_arm_dofs = 6 
        self.num_total_dofs = self.num_arm_dofs + self.num_hand_dofs
        self.n_stack = self.config['task']["env"].get("obs_stack", 4)
        self.n_obs_dim_single_frame = 85 
        
        self.actions_num = self.num_total_dofs 
        self.device = 'cuda'

        self.debug_viz = self.config['task']['env']['enableDebugVis']
        self.rotation_axis = self.config['task']['default_axis'] #
        self.act_moving_average = self.config['task']["env"]["actionsMovingAverage"]
        self.relative_scale = self.config["task"]["env"].get("relScale", 0.5)

        self.spin_axis = {
            'x': torch.tensor([[1.0, 0.0, 0.0]], device=self.device),
            'y': torch.tensor([[0.0, -1.0, 0.0]], device=self.device),
            'z': torch.tensor([[0.0, 0.0, 1.0]], device=self.device)
        }
        self.players = {}
        self.active_player = None

        self.get_dof_limits()
        self.init_hand_qpos_dict = {
            "a_0": 0.0,
            "a_1": 0.0,
            "a_2": 0.0,
            "a_3": 0.0,
            "a_12": 0.0048,
            "a_13": 0.0,
            "a_14": 0.0,
            "a_15": 0.0,
            "a_4": 1.3815,
            "a_5": 0.0868,
            "a_6": 0.1259,
            "a_7": 0.0,
            "a_8": 0.0,
            "a_9": 0.0,
            "a_10": 0.0,
            "a_11": 0.0
        }
        self.HAND_JOINT_ORDER = [
            "a_1", "a_0", "a_2", "a_3",
            "a_12", "a_13", "a_14", "a_15",
            "a_5", "a_4", "a_6", "a_7",
            "a_9", "a_8", "a_10", "a_11"
        ]

        self.default_arm_pos = [0.00, 1.183, -1.541, 3.1416, 2.742, -1.569]
        #self.default_arm_pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.post_init()

        # if self.debug_viz:
        #     self.setup_plot()
        self.target_axis = self.rotation_axis
        self.start_keyboard_listener()
        self.is_rotating = False   
        self.active_key = None

    def post_init(self):
        arm_hand_dof_default_pos = []
        arm_hand_dof_default_vel = []

        for i in range(22):
            if i < 6:
                arm_hand_dof_default_pos.append(self.default_arm_pos[i])
            else:
                arm_hand_dof_default_pos.append(0.0)
            arm_hand_dof_default_vel.append(0.0)

        arm_hand_dof_default_pos[6:] = [self.init_hand_qpos_dict[f"a_{i}"] for i in range(16)]

        self.arm_hand_dof_default_pos = to_torch(arm_hand_dof_default_pos, device=self.device)
        self.arm_hand_dof_default_vel = to_torch(arm_hand_dof_default_vel, device=self.device)

    def real_to_sim(self, values):
        if not hasattr(self, "real_to_sim_indices"):
            self.construct_sim_to_real_transformation()

        return values[:, self.real_to_sim_indices]

    def sim_to_real(self, values):
        if not hasattr(self, "sim_to_real_indices"):
            self.construct_sim_to_real_transformation()
        
        return values[:, self.sim_to_real_indices]

    def construct_sim_to_real_transformation(self):
        self.sim_to_real_indices = [1, 0, 2, 3, 9, 8, 10, 11, 13, 12, 14, 15, 4, 5, 6, 7]
        self.real_to_sim_indices = [1, 0, 2, 3, 12, 13, 14, 15, 5, 4, 6, 7, 9, 8, 10, 11]

    def get_dof_limits(self):

        self.leap_dof_lower = torch.tensor([-1.0470, -0.3140, -0.5060, -0.3660, -1.0470, -0.3140, -0.5060, -0.3660,
         -1.0470, -0.3140, -0.5060, -0.3660, -0.3490, -0.4700, -1.2000, -1.3400]).to(device=self.device)[None, :]
        self.leap_dof_upper = torch.tensor([1.0470, 2.2300, 1.8850, 2.0420, 1.0470, 2.2300, 1.8850, 2.0420, 1.0470,
         2.2300, 1.8850, 2.0420, 2.0940, 2.4430, 1.9000, 1.8800]).to(device=self.device)[None, :]
        #print("LEAP DOF limits(real):", self.leap_dof_lower,"\n", self.leap_dof_upper)

        self.leap_dof_lower = self.real_to_sim(self.leap_dof_lower).squeeze()
        self.leap_dof_upper = self.real_to_sim(self.leap_dof_upper).squeeze()
        #print("LEAP DOF limits(sim):", self.leap_dof_lower,"\n", self.leap_dof_upper)

    def on_key_press(self, key):
        try:
            char = key.char.lower()
            if char in ['x', 'y', 'z']:
                if self.active_key is None:
                    print(f"\n--- Key '{char}' pressed. Locked control. ---")
                    self.active_key = char       
                    self.target_axis = char   
                    self.is_rotating = True    
                elif self.active_key == char:
                    pass
                else:
                    pass
        except AttributeError:
            pass
    
    def on_key_release(self, key):
        try:
            char = key.char.lower()
            if char == self.active_key:
                self.active_key = None    
                self.is_rotating = False     # 停止转动
            else:
                pass
        except AttributeError:
            pass

    def start_keyboard_listener(self):
        listener = keyboard.Listener(
            on_press=self.on_key_press, 
            on_release=self.on_key_release
        )
        listener.daemon = True
        listener.start()

    def deploy(self):
        # import rospy
        from isaacgymenvs.hardware_controller import LeapHand

        # try to set up rospy
        num_obs = (45+16+24) * self.n_stack # ps
        num_obs_single = num_obs // 4 
        leap = LeapHand()
        leap.leap_dof_lower = self.leap_dof_lower.cpu().numpy()
        leap.leap_dof_upper = self.leap_dof_upper.cpu().numpy()
        leap.sim_to_real_indices = self.sim_to_real_indices
        leap.real_to_sim_indices = self.real_to_sim_indices

        # hz = 20
        hz = 10
        self.control_dt = 1 / hz

        print("command to the initial position")
        initial_command = self.arm_hand_dof_default_pos.cpu().numpy()[6:22]
        for _ in range(hz * 2):
            leap.command_joint_position(initial_command)
            obses, _ = leap.poll_joint_position()
            # ros_rate.sleep()self
            time.sleep(self.control_dt)
        print("done")

        obses, _ = leap.poll_joint_position()
        obses = torch.from_numpy(obses.astype(np.float32)).cuda() # shape (16,)
        
        def unscale(x, lower, upper):
            return (2.0 * x - upper - lower) / (upper - lower)
        
        last_obs_buf = torch.zeros((1, self.n_obs_dim_single_frame), device=self.device, dtype=torch.float)
        last_action = torch.zeros(1, self.num_total_dofs, dtype=torch.float, device=self.device)
        
        prev_target = torch.zeros(1, self.num_total_dofs, device=self.device)
        prev_target[0, self.num_arm_dofs:] = obses.clone() 

        unscaled_hand_pos = unscale(obses, self.leap_dof_lower, self.leap_dof_upper)
        last_obs_buf[0, 6:22] = unscaled_hand_pos
        
        unscaled_hand_target = unscale(prev_target[0, self.num_arm_dofs:], self.leap_dof_lower, self.leap_dof_upper)
        last_obs_buf[0, 29:45] = unscaled_hand_target

        tactile_tensor = torch.zeros(16, device=self.device)
        if flag:
            with tactile_lock:
                tactile_tensor = latest_tactile_tensor.to(self.device)

        # tactile sensor data
        last_obs_buf[0, 45:61] = tactile_tensor

        # [61:85]: 旋转轴 (24维)
        last_obs_buf[0, 61:85] = self.spin_axis[self.rotation_axis].repeat(1, 8)

        obs_buf = last_obs_buf.repeat(1, self.n_stack)

        if self.active_player.is_rnn:
            self.active_player.init_rnn()

        counter = 0 
        self.real_data_log = [] 
        record_steps = 499

        transition_counter = 0
        TRANSITION_STEPS = 50 

        while True:
        # for i in range(len(sim_targets)):
            loop_start_time = time.perf_counter()

            if self.target_axis != self.rotation_axis:
                print(f"\nSwitching model from axis '{self.rotation_axis}' to '{self.target_axis}'...")
                self.rotation_axis = self.target_axis 
                self.active_player = self.players[self.rotation_axis]
                
                if self.active_player.is_rnn:
                    self.active_player.init_rnn() 
                
                obses, _ = leap.poll_joint_position()
                obses = torch.from_numpy(obses.astype(np.float32)).cuda()
                prev_target[0, self.num_arm_dofs:] = obses.clone()

                unscaled_hand_pos = unscale(obses, self.leap_dof_lower, self.leap_dof_upper)
                last_obs_buf[0, 6:22] = unscaled_hand_pos
                last_obs_buf[0, 29:45] = unscaled_hand_pos 
                tactile_tensor = torch.zeros(16, device=self.device)
                if flag:
                    with tactile_lock:
                        tactile_tensor = latest_tactile_tensor.to(self.device)
                last_obs_buf[0, 45:61] = tactile_tensor
                last_obs_buf[0, 61:85] = self.spin_axis[self.rotation_axis].repeat(1, 8)
                obs_buf = last_obs_buf.repeat(1, self.n_stack)

                last_action.zero_()

                transition_counter = TRANSITION_STEPS
                
                print(f"Model for axis '{self.rotation_axis}' is now active.")

            counter += 1
            # obs = self.running_mean_std(obs_buf.clone()) # ! Need to check if this is implemented
            
            if self.is_rotating == True:
                if hasattr(self, "actions_list"):
                    action = self.actions_list[counter-1][None, :]
                else:
                    action = self.forward_network(obs_buf)
            else:
                action = torch.zeros((1, self.num_total_dofs), device=self.device)

            action = torch.clamp(action, -1.0, 1.0)

            #ramp up action from zero when switching models
            # if transition_counter > 0:
            #     transition_counter -= 1
            #     scale_factor = 1.0 - transition_counter / TRANSITION_STEPS
            #     action = action * scale_factor

            if "actions_mask" in self.config["task"]["env"]:
                action = action * torch.tensor(self.config["task"]["env"]["actions_mask"]).cuda()[None, :]

            action = action * self.act_moving_average + last_action * (1.0 - self.act_moving_average)
            # action = action * 0.8 + last_action * (1.0 - 0.8)
            
            # ran_relative_scale = self.relative_scale * (1 + (torch.rand(1,1,device=self.device) - 0.5) * 0.1)
            # target = prev_target + ran_relative_scale * action 
            target = prev_target + self.relative_scale * action
            #target[0, self.num_arm_dofs:] += torch.tensor(adjustment, device=self.device)
            target[0, self.num_arm_dofs:] = torch.clamp(
                target[0, self.num_arm_dofs:], 
                self.leap_dof_lower, 
                self.leap_dof_upper
            )
            prev_target = target.clone()
            last_action = action.clone()
        
            # interact with the hardware
            commands = target[0, self.num_arm_dofs:].cpu().numpy()

            if "disable_actions" not in self.config['task']["env"]:
                leap.command_joint_position(commands)

            # maintain the loop rate
            elapsed_time = time.perf_counter() - loop_start_time  
            sleep_time = self.control_dt - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)  
            
            # get o_{t+1}
            obses, _ = leap.poll_joint_position()
            obses = torch.from_numpy(obses.astype(np.float32)).cuda() # (16,)

            tactile_tensor = torch.zeros(16, device=self.device)
            if flag:
                with tactile_lock:
                    tactile_tensor = latest_tactile_tensor.to(self.device)
            # print("tactile_tensor:", tactile_tensor)
            last_obs_buf[0, 0:6] = 0.0
            last_obs_buf[0, 6:22] = unscale(obses, self.leap_dof_lower, self.leap_dof_upper)
            last_obs_buf[0, 22:29] = 0.0
            last_obs_buf[0, 29:45] = unscale(prev_target[0, self.num_arm_dofs:], self.leap_dof_lower, self.leap_dof_upper)
            last_obs_buf[0, 45:61] = tactile_tensor
            last_obs_buf[0, 61:85] = self.spin_axis[self.rotation_axis].repeat(1, 8)
            obs_buf = torch.cat((last_obs_buf, obs_buf[:, :-self.n_obs_dim_single_frame]), dim=-1)

            obs_buf = obs_buf.float()

    def forward_network(self, obs):
        return self.active_player.get_action(obs, True)
    
    def restore_all_models(self): 
        print("Loading all rotation models...")
        for axis, model_config in self.config['task']['models'].items():
            print(f"  -> Loading model for axis: '{axis}' from {model_config['checkpoint']}")
            
            rlg_config_dict = copy.deepcopy(self.config['train']) 
            rlg_config_dict["params"]["config"]["env_info"] = {}
            self.num_obs = (45+16+24) * self.n_stack
            self.num_actions = 22
            
            observation_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
            rlg_config_dict["params"]["config"]["env_info"]["observation_space"] = observation_space
            action_space = spaces.Box(np.ones(self.num_actions) * -1., np.ones(self.num_actions) * 1.)
            rlg_config_dict["params"]["config"]["env_info"]["action_space"] = action_space
            rlg_config_dict["params"]["config"]["env_info"]["agents"] = 1

            def build_runner(algo_observer):
                runner = Runner(algo_observer)
                runner.algo_factory.register_builder('amp_continuous', lambda **kwargs : amp_continuous.AMPAgent(**kwargs))
                runner.player_factory.register_builder('amp_continuous', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))
                model_builder.register_model('continuous_amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))
                model_builder.register_network('amp', lambda **kwargs : amp_network_builder.AMPBuilder())
                return runner

            runner = build_runner(RLGPUAlgoObserver())
            runner.load(rlg_config_dict)
            runner.reset()

            args = {
                'train': False,
                'play': True,
                'checkpoint' : model_config['checkpoint'], 
                'sigma' : None
            }

            player = runner.create_player()
            _restore(player, args)
            _override_sigma(player, args)
            
            self.players[axis] = player

        self.active_player = self.players[self.rotation_axis]
        print(f"\nAll models loaded. Default active model is for axis: '{self.rotation_axis}'")

    # def restore(self):
    #     rlg_config_dict = self.config['train']
    #     rlg_config_dict["params"]["config"]["env_info"] = {}
    #     self.num_obs = (45+16+24) * self.n_stack  # ps
    #     self.num_actions = 22
    #     # 检查观测和动作空间的维度
    #     # ----------------------------------
    #     expected_num_obs = self.n_obs_dim_single_frame * self.n_stack
    #     if self.num_obs != expected_num_obs:
    #         print(f"Warning: numObservations in config ({self.num_obs}) does not match expected ({expected_num_obs})!")
    #     if self.num_actions != self.num_total_dofs:
    #         print(f"Warning: numActions in config ({self.num_actions}) does not match expected ({self.num_total_dofs})!")
    #     # ----------------------------------
    #     observation_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
    #     rlg_config_dict["params"]["config"]["env_info"]["observation_space"] = observation_space
    #     action_space = spaces.Box(np.ones(self.num_actions) * -1., np.ones(self.num_actions) * 1.)
    #     rlg_config_dict["params"]["config"]["env_info"]["action_space"] = action_space
    #     rlg_config_dict["params"]["config"]["env_info"]["agents"] = 1

    #     def build_runner(algo_observer):
    #         runner = Runner(algo_observer)
    #         runner.algo_factory.register_builder('amp_continuous', lambda **kwargs : amp_continuous.AMPAgent(**kwargs))
    #         runner.player_factory.register_builder('amp_continuous', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))
    #         model_builder.register_model('continuous_amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))
    #         model_builder.register_network('amp', lambda **kwargs : amp_network_builder.AMPBuilder())

    #         return runner

    #     runner = build_runner(RLGPUAlgoObserver())
    #     runner.load(rlg_config_dict)
    #     runner.reset()

    #     args = {
    #         'train': False,
    #         'play': True,
    #         'checkpoint' : self.config['checkpoint'],
    #         'sigma' : None
    #     }

    #     self.player = runner.create_player()
    #     _restore(self.player, args)
    #     _override_sigma(self.player, args)


@hydra.main(config_name='config', config_path='cfg')
def main(config: DictConfig):
    agent = HardwarePlayer(config)
    #agent.restore()
    agent.restore_all_models()
    agent.deploy()

if __name__ == '__main__':
    main()