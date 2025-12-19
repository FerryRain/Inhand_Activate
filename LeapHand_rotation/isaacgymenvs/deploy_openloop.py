#!/usr/bin/env python3
# --------------------------------------------------------
# LEAP Hand Open-Loop Replay (With Manual Stop)
# --------------------------------------------------------

import sys
import os

# Fix path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from isaacgymenvs.hardware_controller import LeapHand
from pynput import keyboard  # <--- 新增库

# ================= Configuration =================
SIM_DATA_PATH = "runs/rollout/z_ps/data_part_0.npz"
SAVE_PATH = "real_world_dataset.npz"

CONTROL_HZ = 10.0         
RELATIVE_SCALE = 0.2      
ACT_MOVING_AVERAGE = 0.8  
OBS_DIM = 85              

# 1. 原始的物理限制 (Real Order)
RAW_LEAP_LOWER = np.array([-1.047, -0.314, -0.506, -0.366, -1.047, -0.314, -0.506, -0.366,
                           -1.047, -0.314, -0.506, -0.366, -0.349, -0.470, -1.200, -1.340])
RAW_LEAP_UPPER = np.array([1.047, 2.230, 1.885, 2.042, 1.047, 2.230, 1.885, 2.042,
                           1.047, 2.230, 1.885, 2.042, 2.094, 2.443, 1.900, 1.880])

# 2. 关节映射索引
REAL_TO_SIM_INDICES = [1, 0, 2, 3, 12, 13, 14, 15, 5, 4, 6, 7, 9, 8, 10, 11]
SIM_TO_REAL_INDICES = [1, 0, 2, 3, 9, 8, 10, 11, 13, 12, 14, 15, 4, 5, 6, 7]

init_hand_qpos_override_dict = {
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
Defaut_pos = [init_hand_qpos_override_dict[f"a_{i}"] for i in range(16)]

# =================================================

class ReplayAgent:
    def __init__(self):
        print("[HW] Initializing LeapHand Interface...")
        self.leap = LeapHand()
        
        self.leap.sim_to_real_indices = SIM_TO_REAL_INDICES
        self.leap.real_to_sim_indices = REAL_TO_SIM_INDICES
        
        self.lower_sim = RAW_LEAP_LOWER[REAL_TO_SIM_INDICES].squeeze()
        self.upper_sim = RAW_LEAP_UPPER[REAL_TO_SIM_INDICES].squeeze()
        
        self.leap.leap_dof_lower = self.lower_sim
        self.leap.leap_dof_upper = self.upper_sim
        
        self.dt = 1.0 / CONTROL_HZ
        
        self.trajectories = self._load_npz(SIM_DATA_PATH)
        self.real_data = {'obs': [], 'acts': [], 'dones': []}
        self.collected_count = 0

        # === 新增：键盘监听状态 ===
        self.stop_triggered = False
        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start() # 启动后台线程监听键盘

    def _on_press(self, key):
        """键盘回调函数"""
        try:
            # 检测空格键
            if key == keyboard.Key.space:
                self.stop_triggered = True
        except AttributeError:
            pass

    def _load_npz(self, path):
        if not os.path.exists(path):
            print(f"Error: File {path} not found.")
            sys.exit(1)
        print(f"[Data] Loading {path}...")
        data = np.load(path)
        obs, acts, dones = data['obs'], data['acts'], data['dones']
        
        trajs = []
        curr_a = []
        for i in range(len(acts)):
            curr_a.append(acts[i])
            if dones[i]:
                if len(curr_a) > 50:
                    trajs.append(np.array(curr_a))
                curr_a = []
        print(f"[Data] Loaded {len(trajs)} valid trajectories.")
        return trajs

    def unscale(self, x, lower, upper):
        return (2.0 * x - upper - lower) / (upper - lower)

    def reset_hand(self):
        print("[Ctrl] Resetting Hand...")
        initial_command = np.array(Defaut_pos, dtype=np.float32)
        for _ in range(int(CONTROL_HZ * 2.0)):
            self.leap.command_joint_position(initial_command)
            time.sleep(self.dt)

    def run(self):
        print("\n" + "="*50)
        print(f" OPEN-LOOP REPLAY (Spacebar to EMERGENCY STOP)")
        print("="*50)
        
        import random
        random.seed(time.time())
        indices = list(range(len(self.trajectories)))
        random.shuffle(indices)

        for i, idx in enumerate(indices):
            raw_actions_traj = self.trajectories[idx]
            steps = len(raw_actions_traj)
            print(f"\n>>> Trial {i+1} (Traj ID: {idx}, Steps: {steps})")
            
            # 1. Reset
            self.reset_hand()
            
            # 2. Wait
            print("\n>>> Please place the PEN.")
            cmd = input(">>> Press [Enter] to start, [s] to skip, [q] to quit: ")
            if cmd.lower() == 'q': break
            if cmd.lower() == 's': continue

            print(">>> Replaying... (Press SPACE to Stop)")
            
            # 重置停止标志位
            self.stop_triggered = False

            # --- Initialization ---
            curr_pos_sim, _ = self.leap.poll_joint_position()
            prev_target_sim = curr_pos_sim.copy()
            last_action = np.zeros(16, dtype=np.float32)

            trial_obs = []
            trial_acts = []
            
            # 标记是否被手动停止
            aborted = False 

            for t in range(steps):
                # === 新增：检测是否按下空格 ===
                if self.stop_triggered:
                    print("\n[Ctrl] !!! MANUAL STOP TRIGGERED !!!")
                    aborted = True
                    break
                # ===========================

                t0 = time.perf_counter()
                
                # ... (常规计算逻辑) ...
                raw_act_full = raw_actions_traj[t]
                if raw_act_full.shape[0] == 22:
                    action = raw_act_full[6:]
                else:
                    action = raw_act_full
                
                filtered_action = action * ACT_MOVING_AVERAGE + last_action * (1.0 - ACT_MOVING_AVERAGE)
                target_sim = prev_target_sim + RELATIVE_SCALE * filtered_action
                target_sim = np.clip(target_sim, self.lower_sim, self.upper_sim)
                
                prev_target_sim = target_sim.copy()
                last_action = filtered_action.copy()
                
                self.leap.command_joint_position(target_sim)
                
                curr_pos_sim, _ = self.leap.poll_joint_position()
                
                obs_vec = np.zeros(OBS_DIM, dtype=np.float32)
                obs_vec[6:22] = self.unscale(curr_pos_sim, self.lower_sim, self.upper_sim)
                obs_vec[29:45] = self.unscale(prev_target_sim, self.lower_sim, self.upper_sim)
                
                act_vec = np.zeros(22, dtype=np.float32)
                act_vec[6:] = filtered_action
                
                trial_obs.append(obs_vec)
                trial_acts.append(act_vec)
                
                dt = time.perf_counter() - t0
                if dt < self.dt:
                    time.sleep(self.dt - dt)
            
            # Post-Trial
            if aborted:
                print("    -> Trial Aborted by User.")
                valid = input(">>> Was it somehow SUCCESSFUL? [y/n] (default n): ")
            else:
                valid = input(">>> SUCCESSFUL? [y/n]: ")
            
            if valid.lower() == 'y':
                print(f"    -> Saved {len(trial_obs)} steps.")
                self.real_data['obs'].extend(trial_obs)
                self.real_data['acts'].extend(trial_acts)
                d = np.zeros(len(trial_obs), dtype=bool)
                d[-1] = True
                self.real_data['dones'].extend(d)
                self.collected_count += 1
                if self.collected_count % 5 == 0: 
                    self._save(f"real_checkpoint_{self.collected_count}.npz")
            else:
                print("    -> Discarded.")

        self._save(SAVE_PATH)

    def _save(self, filename):
        if len(self.real_data['obs']) == 0: return
        print(f"\n[IO] Saving to {filename}...")
        np.savez_compressed(
            filename, 
            obs=np.array(self.real_data['obs']), 
            acts=np.array(self.real_data['acts']), 
            dones=np.array(self.real_data['dones'])
        )
        print("[IO] Saved.")

if __name__ == "__main__":
    try:
        agent = ReplayAgent()
        agent.run()
    except KeyboardInterrupt:
        print("\n[Ctrl] Interrupted.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[Error] {e}")