import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# ================= Configuration =================
# 指向你刚刚用 train.py 收集的数据文件
DATA_PATH = "runs/rollout/teacher_data_z_ps_filtered/data_part_0.npz" 

# 与 Deploy 代码保持完全一致的参数
CONTROL_HZ = 10.0         
RELATIVE_SCALE = 0.2      
ACT_MOVING_AVERAGE = 0.8  

# 关节限制 (必须与 Deploy 代码中的顺序和数值一致)
# 原始 Real Order 限制
RAW_LEAP_LOWER = np.array([-1.047, -0.314, -0.506, -0.366, -1.047, -0.314, -0.506, -0.366,
                           -1.047, -0.314, -0.506, -0.366, -0.349, -0.470, -1.200, -1.340])
RAW_LEAP_UPPER = np.array([1.047, 2.230, 1.885, 2.042, 1.047, 2.230, 1.885, 2.042,
                           1.047, 2.230, 1.885, 2.042, 2.094, 2.443, 1.900, 1.880])
# 映射索引
REAL_TO_SIM_INDICES = [1, 0, 2, 3, 12, 13, 14, 15, 5, 4, 6, 7, 9, 8, 10, 11]

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

# 计算 Sim 顺序下的限制
LOWER_LIMITS = RAW_LEAP_LOWER[REAL_TO_SIM_INDICES]
UPPER_LIMITS = RAW_LEAP_UPPER[REAL_TO_SIM_INDICES]

# =================================================

def load_data(path):
    if not os.path.exists(path):
        print(f"Error: File {path} not found.")
        return None
    data = np.load(path)
    # 确保 targets 存在
    if 'targets' not in data:
        print("Error: 'targets' key not found in npz. Please re-collect data using the modified train.py.")
        return None
    return data

def process_trajectory(obs, acts, gt_targets):
    """
    使用 Deploy 的逻辑重新计算 Target
    """
    steps = len(acts)
    calculated_targets = []
    
    # 方案 B: 使用代码中定义的固定 Default_pos 作为起点
    # 如果仿真数据的起点与 Default_pos 不同，曲线会出现整体平移
    current_target = np.array(Defaut_pos, dtype=np.float32).copy()
    #current_target = gt_targets[0].copy()  # 也可以用轨迹的第一个 Target 作为起点
    
    last_action = np.zeros(16, dtype=np.float32) # Deploy 中初始化为 0
    
    for t in range(steps):
        # 1. 获取 Action (处理维度 22 vs 16)
        raw_act_full = acts[t]
        if raw_act_full.shape[0] == 22:
            action = raw_act_full[6:] # 切片掉机械臂的 6 个维度
        else:
            action = raw_act_full
            
        # 2. 移动平均 (Filter)
        filtered_action = action * ACT_MOVING_AVERAGE + last_action * (1.0 - ACT_MOVING_AVERAGE)
        last_action = filtered_action.copy()
        
        # 3. 相对控制积分 (Integration)
        next_target = current_target + RELATIVE_SCALE * filtered_action
        
        # 4. 限位 (Clip)
        next_target = np.clip(next_target, LOWER_LIMITS, UPPER_LIMITS)
        
        calculated_targets.append(next_target.copy())
        
        # 更新 current_target 用于下一步
        current_target = next_target.copy()
        
    return np.array(calculated_targets)

def main():
    data = load_data(DATA_PATH)
    if data is None: return

    full_obs = data['obs']
    full_acts = data['acts']
    full_targets = data['targets']
    full_dones = data['dones']

    # 分割轨迹：找到所有 done 为 True 的位置作为分割点
    trajectory_starts = [0] + (np.where(full_dones)[0] + 1).tolist()
    
    total_trajs = len(trajectory_starts) - 1
    print(f"Found {total_trajs} trajectories in total.")

    # 循环遍历所有轨迹
    for traj_idx in range(total_trajs):
        start_idx = trajectory_starts[traj_idx]
        end_idx = trajectory_starts[traj_idx + 1] if traj_idx + 1 < len(trajectory_starts) else len(full_acts)
        
        if end_idx - start_idx < 10:
            continue

        print(f"\n" + "="*50)
        print(f"Analyzing Trajectory {traj_idx}/{total_trajs} (Steps: {end_idx - start_idx})...")

        traj_obs = full_obs[start_idx:end_idx]
        traj_acts = full_acts[start_idx:end_idx]
        raw_traj_targets = full_targets[start_idx:end_idx]

        contact_data = traj_obs[:,45:61]
        
        # 提取手的 Target (最后16维)
        if raw_traj_targets.shape[1] == 22:
            gt_targets = raw_traj_targets[:, 6:]
        else:
            gt_targets = raw_traj_targets

        # ==========================================
        # [新增] 打印初始 Target
        # ==========================================
        print("\n[Start Info]")
        # 打印仿真数据中真实的起始 Target
        print("  Real Sim Start Target (GT) :")
        with np.printoptions(precision=4, suppress=True):
            print(f"    {gt_targets[0]}")
        
        # 打印脚本使用的计算起点 (Default Pos)
        print("  Calculated Start (Defaut)  :")
        with np.printoptions(precision=4, suppress=True):
            print(f"    {np.array(Defaut_pos)}")
            
        # 计算起点误差
        start_diff = np.abs(gt_targets[0] - np.array(Defaut_pos))
        if np.any(start_diff > 1e-3):
            print(f"  [WARNING] Start Position Mismatch! Max Diff: {np.max(start_diff):.4f}")
            print(f"  (This will cause constant offset in the plot)")
        else:
            print("  [OK] Start Positions Match.")
        # ==========================================

        calc_targets = process_trajectory(traj_obs, traj_acts, gt_targets)

        diff_per_joint = np.mean(np.abs(gt_targets - calc_targets), axis=0)
        max_err_joint = np.argmax(diff_per_joint)
        mean_err = np.mean(diff_per_joint)
        
        print(f"\n[Error Stats]")
        print(f"  Mean Error across all joints: {mean_err:.6f}")
        print(f"  Max Error Joint Index: {max_err_joint} (Error: {diff_per_joint[max_err_joint]:.6f})")

        # === 绘图 ===
        joints_to_plot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, max_err_joint]
        joints_to_plot = sorted(list(set(joints_to_plot))) # 去重

        fig, axes = plt.subplots(len(joints_to_plot), 1, figsize=(10, 2 * len(joints_to_plot)), sharex=True)
        if len(joints_to_plot) == 1: axes = [axes]

        steps = np.arange(len(calc_targets))

        for i, joint_idx in enumerate(joints_to_plot):
            ax = axes[i]
            # 画 Sim 真实 Target
            ax.plot(steps, gt_targets[:, joint_idx], 'k-', linewidth=2, label='Sim Target' if i==0 else "")
            # 画 计算推演 Target
            ax.plot(steps, calc_targets[:, joint_idx], 'r--', linewidth=2, label='Calc Target' if i==0 else "")
            
            title_str = f"Joint {joint_idx}"
            if joint_idx == max_err_joint:
                title_str += " (MAX ERROR)"
            ax.set_ylabel(title_str, rotation=0, labelpad=50)
            ax.grid(True, alpha=0.3)

        # 只在第一张图显示 Legend
        axes[0].legend(loc="upper right")
        axes[-1].set_xlabel("Time Step")
        
        fig.suptitle(f"Trajectory {traj_idx} - Press 'q' to Quit, Any other key for Next", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # === 交互控制逻辑 ===
        # 定义按键回调函数
        def on_key(event):
            if event.key == 'q':
                print("Quit key pressed. Exiting...")
                plt.close(fig)
                sys.exit(0)
            else:
                # 按其他键，关闭当前图，循环继续到下一张
                plt.close(fig)

        # 绑定事件并显示
        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show() # 这里会阻塞，直到 plt.close(fig) 被调用

if __name__ == "__main__":
    main()