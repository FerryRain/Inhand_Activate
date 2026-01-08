import sys
from types import ModuleType
import torch
if "isaacgym" not in sys.modules:
    ig = ModuleType("isaacgym")
    sys.modules["isaacgym"] = ig
    
    tu = ModuleType("isaacgym.torch_utils")
    tu.to_torch = lambda x, dtype=None, device='cuda:0', **kwargs: torch.as_tensor(x, dtype=dtype, device=device)
    tu.torch_rand_float = lambda low, high, size, device: (low - high) * torch.rand(*size, device=device) + high
    
    sys.modules["isaacgym.torch_utils"] = tu
    ig.torch_utils = tu


import os
import hydra
from omegaconf import DictConfig, OmegaConf
import sys
import numpy as np
import time
import serial
import threading
import cv2
import copy
from scipy.ndimage import gaussian_filter
from pynput import keyboard
from gym import spaces
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_games.torch_runner import Runner, _override_sigma, _restore
from rl_games.algos_torch import model_builder
from isaacgymenvs.learning import amp_continuous, amp_models, amp_network_builder, amp_players
from isaacgymenvs.hardware_controller import LeapHand 

try:
    from torch.serialization import add_safe_globals
    # 兼容 numpy 1.x 和 2.x
    import numpy.core.multiarray as _ncm
    add_safe_globals([_ncm.scalar])
except Exception:
    pass

def to_torch(x, device='cuda'):
    return torch.as_tensor(x, dtype=torch.float32, device=device)

def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)

contact_data_norm = np.zeros((16,16))
THRESHOLD = 5
NOISE_SCALE = 5
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
        self.config = OmegaConf.to_container(config, resolve=True)
        
        self.num_hand_dofs = 16
        self.num_arm_dofs = 6 
        self.num_total_dofs = self.num_arm_dofs + self.num_hand_dofs
        self.n_stack = self.config['task']["env"].get("obs_stack", 4)
        self.n_obs_dim_single_frame = 85 
        self.device = 'cuda'

        self.rotation_axis = self.config['task']['default_axis']
        self.act_moving_average = self.config['task']["env"]["actionsMovingAverage"]
        self.relative_scale = self.config["task"]["env"].get("relScale", 0.5)

        self.spin_axis = {
            'x': torch.tensor([[1.0, 0.0, 0.0]], device=self.device),
            'y': torch.tensor([[0.0, -1.0, 0.0]], device=self.device),
            'z': torch.tensor([[0.0, 0.0, 1.0]], device=self.device)
        }
        self.players = {}
        self.active_player = None

        self.construct_sim_to_real_transformation()
        self.get_dof_limits()
        
        self.init_hand_qpos_dict = {
            "a_0": 0.0,     "a_1": 0.0,     "a_2": 0.0,     "a_3": 0.0,
            "a_12": 0.0048, "a_13": 0.0,    "a_14": 0.0,    "a_15": 0.0,
            "a_4": 1.3815,  "a_5": 0.0868,  "a_6": 0.1259,  "a_7": 0.0,
            "a_8": 0.0,     "a_9": 0.0,     "a_10": 0.0,    "a_11": 0.0
        }
        self.default_arm_pos = [0.00, 1.183, -1.541, 3.1416, 2.742, -1.569]
        
        self.post_init()
        self.target_axis = self.rotation_axis
        self.start_keyboard_listener()
        self.is_rotating = False   
        self.active_key = None

    def post_init(self):
        arm_hand_dof_default_pos = []
        for i in range(22):
            if i < 6:
                arm_hand_dof_default_pos.append(self.default_arm_pos[i])
            else:
                arm_hand_dof_default_pos.append(self.init_hand_qpos_dict[f"a_{i-6}"])
        
        self.arm_hand_dof_default_pos = to_torch(arm_hand_dof_default_pos, device=self.device)

    def construct_sim_to_real_transformation(self):
        self.sim_to_real_indices = [1, 0, 2, 3, 9, 8, 10, 11, 13, 12, 14, 15, 4, 5, 6, 7]
        self.real_to_sim_indices = [1, 0, 2, 3, 12, 13, 14, 15, 5, 4, 6, 7, 9, 8, 10, 11]

    def real_to_sim(self, values):
        return values[:, self.real_to_sim_indices]

    def get_dof_limits(self):
        lower = [-1.0470, -0.3140, -0.5060, -0.3660, -1.0470, -0.3140, -0.5060, -0.3660,
                 -1.0470, -0.3140, -0.5060, -0.3660, -0.3490, -0.4700, -1.2000, -1.3400]
        upper = [1.0470, 2.2300, 1.8850, 2.0420, 1.0470, 2.2300, 1.8850, 2.0420, 1.0470,
                 2.2300, 1.8850, 2.0420, 2.0940, 2.4430, 1.9000, 1.8800]
        
        self.leap_dof_lower = self.real_to_sim(to_torch(lower, device=self.device).unsqueeze(0)).squeeze()
        self.leap_dof_upper = self.real_to_sim(to_torch(upper, device=self.device).unsqueeze(0)).squeeze()

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
                self.is_rotating = False    
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
        leap = LeapHand()
        leap.leap_dof_lower = self.leap_dof_lower.cpu().numpy()
        leap.leap_dof_upper = self.leap_dof_upper.cpu().numpy()
        leap.sim_to_real_indices = self.sim_to_real_indices
        leap.real_to_sim_indices = self.real_to_sim_indices

        hz = 10
        self.control_dt = 1 / hz

        print("Commanding to the initial position...")
        initial_command = self.arm_hand_dof_default_pos.cpu().numpy()[6:22]
        for _ in range(hz * 2):
            leap.command_joint_position(initial_command)
            time.sleep(self.control_dt)

        obses, _ = leap.poll_joint_position()
        obses = to_torch(obses, device=self.device)
        
        last_obs_buf = torch.zeros((1, self.n_obs_dim_single_frame), device=self.device)
        last_action = torch.zeros(1, self.num_total_dofs, device=self.device)
        prev_target = torch.zeros(1, self.num_total_dofs, device=self.device)
        prev_target[0, 6:] = obses.clone() 

        last_obs_buf[0, 6:22] = unscale(obses, self.leap_dof_lower, self.leap_dof_upper)
        last_obs_buf[0, 29:45] = last_obs_buf[0, 6:22].clone()
        
        obs_buf = last_obs_buf.repeat(1, self.n_stack)

        if self.active_player.is_rnn:
            self.active_player.init_rnn()

        print("Start deployment loop...")
        while True:
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
                
                print(f"Model for axis '{self.rotation_axis}' is now active.")

            if self.is_rotating:
                action = self.active_player.get_action(obs_buf, True)
            else:
                action = torch.zeros((1, self.num_total_dofs), device=self.device)

            action = torch.clamp(action, -1.0, 1.0)
            
            action = action * self.act_moving_average + last_action * (1.0 - self.act_moving_average)
            target = prev_target + self.relative_scale * action
            target[0, 6:] = torch.clamp(target[0, 6:], self.leap_dof_lower, self.leap_dof_upper)
            
            leap.command_joint_position(target[0, 6:].cpu().numpy())

            obses, _ = leap.poll_joint_position()
            obses = to_torch(obses, device=self.device)
  
            last_obs_buf.fill_(0)
            last_obs_buf[0, 6:22] = unscale(obses, self.leap_dof_lower, self.leap_dof_upper)
            last_obs_buf[0, 29:45] = unscale(target[0, 6:], self.leap_dof_lower, self.leap_dof_upper)
            with tactile_lock:
                last_obs_buf[0, 45:61] = latest_tactile_tensor.to(self.device)
            last_obs_buf[0, 61:85] = self.spin_axis[self.rotation_axis].repeat(1, 8)
            
            obs_buf = torch.cat((last_obs_buf, obs_buf[:, :-self.n_obs_dim_single_frame]), dim=-1)
            
            prev_target = target.clone()
            last_action = action.clone()

            sleep_time = self.control_dt - (time.perf_counter() - loop_start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def restore_all_models(self): 
        model_builder.register_model('continuous_amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))
        model_builder.register_network('amp', lambda **kwargs : amp_network_builder.AMPBuilder())
        
        for axis, model_config in self.config['task']['models'].items():
            print(f"Loading model for axis: {axis}...")
            
            rlg_config = copy.deepcopy(self.config['train'])
            rlg_config["params"]["config"]["env_info"] = {
                "observation_space": spaces.Box(-np.inf, np.inf, (self.n_obs_dim_single_frame * self.n_stack,)),
                "action_space": spaces.Box(-1.0, 1.0, (self.num_total_dofs,)),
                "agents": 1
            }

            runner = Runner()
 
            runner.player_factory.register_builder('amp_continuous', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))
            
            runner.load(rlg_config)
            runner.reset()
            
            player = runner.create_player()
            _restore(player, {'checkpoint': model_config['checkpoint']})
            _override_sigma(player, {'sigma': None})
            
            self.players[axis] = player

        self.active_player = self.players[self.rotation_axis]

@hydra.main(config_name='config', config_path='cfg')
def main(config: DictConfig):
    agent = HardwarePlayer(config)
    agent.restore_all_models()
    agent.deploy()

if __name__ == '__main__':
    main()