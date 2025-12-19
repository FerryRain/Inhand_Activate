import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# 1. 创建保存图片的文件夹
output_dir = "comparison_plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

# 2. 读取数据
try:
    df_sim = pd.read_csv("isaacgymenvs/sim_joint_data.csv")
    df_real = pd.read_csv("isaacgymenvs/real_joint_data.csv")
    print("Data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

# 3. 设置参数
# Sim 结构: [Arm(6) + Hand(16)] -> Hand starts at index 6
# Real 结构: [Hand(16)] -> Hand starts at index 0
sim_offset = 6 
num_hand_joints = 16
plot_steps = min(len(df_sim), len(df_real), 500) # 限制绘图步数，取两者较短者或前500步

print(f"Plotting for {plot_steps} steps...")

# 4. 循环绘制每个关节
for i in range(num_hand_joints):
    # 计算对应的列索引
    sim_idx = i + sim_offset  # Sim 中手部关节从 6 开始
    real_idx = i              # Real 中手部关节从 0 开始
    
    # 获取列名 (根据你之前的保存代码生成的列名)
    col_sim_pos = f"j_pos_{sim_idx}"
    col_real_pos = f"j_pos_{real_idx}"
    
    # 动作索引通常在 Sim 和 Real CSV 中是一样的 (因为记录的是网络输出的全量动作)
    # 网络输出的第 6 个动作对应手部的第 0 个关节
    col_action = f"action_{sim_idx}" 
    
    # 检查列是否存在，防止报错
    if col_sim_pos not in df_sim.columns or col_real_pos not in df_real.columns:
        print(f"Skipping Joint {i}: Column not found.")
        continue

    # 创建画布：2行1列
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # --- 子图 1: 关节角度对比 ---
    ax_pos = axes[0]
    ax_pos.plot(df_sim[col_sim_pos].iloc[:plot_steps], label=f"Sim Joint {sim_idx}", color='blue', alpha=0.7, linewidth=1.5)
    ax_pos.plot(df_real[col_real_pos].iloc[:plot_steps], label=f"Real Joint {real_idx}", color='orange', alpha=0.7, linewidth=1.5)
    ax_pos.set_title(f"Hand Joint {i} (Sim Idx {sim_idx} vs Real Idx {real_idx}) - Position")
    ax_pos.set_ylabel("Joint Position (Rad)")
    ax_pos.grid(True, linestyle='--', alpha=0.6)
    ax_pos.legend()

    # --- 子图 2: 策略动作输出对比 ---
    ax_act = axes[1]
    # 检查动作列是否存在
    if col_action in df_sim.columns and col_action in df_real.columns:
        ax_act.plot(df_sim[col_action].iloc[:plot_steps], label=f"Sim Action {sim_idx}", color='green', alpha=0.7)
        ax_act.plot(df_real[col_action].iloc[:plot_steps], label=f"Real Action {sim_idx}", color='red', alpha=0.7)
        ax_act.set_title(f"Hand Joint {i} - Policy Action Output")
        ax_act.set_ylabel("Action Value")
        ax_act.set_xlabel("Steps")
        ax_act.grid(True, linestyle='--', alpha=0.6)
        ax_act.legend()
    else:
        ax_act.text(0.5, 0.5, "Action data not found", ha='center')

    # 调整布局并保存
    # plt.tight_layout()
    save_path = os.path.join(output_dir, f"joint_{i}_comparison.png")
    plt.savefig(save_path)
    plt.close(fig) # 关闭图表释放内存
    
    print(f"Saved: {save_path}")

print("All plots saved!")