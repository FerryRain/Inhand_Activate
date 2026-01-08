import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

def plot_simulation_data(csv_file="isaacgymenvs/real_joint_data.csv"):
    try:
        df = pd.read_csv(csv_file)
        print(f"成功读取文件: {csv_file}, 共 {len(df)} 步数据。")
    except FileNotFoundError:
        print(f"错误: 找不到文件 {csv_file}。请先运行仿真生成数据。")
        return

    # ==========================================
    # 1. 关节位置跟踪图 (Joint Position vs Target)
    # ==========================================
    
    # 自动识别关节列和目标列
    pos_cols = [c for c in df.columns if c.startswith('j_pos_')]
    target_cols = [c for c in df.columns if c.startswith('target_')]
    
    num_joints = len(pos_cols)
    print(f"检测到 {num_joints} 个关节。")

    # 动态计算子图布局 (例如 22 个关节用 4列 x 6行)
    cols = 4
    rows = math.ceil(num_joints / cols)
    
    fig_joints, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows), sharex=True)
    fig_joints.suptitle(f'Joint Position Tracking (Blue) vs Target (Red)', fontsize=16)
    axes = axes.flatten()

    steps = range(len(df))

    for i in range(num_joints):
        ax = axes[i]
        col_name = pos_cols[i]
        
        # 尝试匹配对应的 target 列
        target_name = f"target_{i}"
        
        # 绘制实际位置
        ax.plot(steps, df[col_name], label='Actual Pos', color='tab:blue', linewidth=1.5)
        
        # 绘制目标位置 (如果有)
        if target_name in df.columns:
            ax.plot(steps, df[target_name], label='Target', color='tab:red', linestyle='--', linewidth=1.5, alpha=0.8)
        
        ax.set_title(f'Joint {i}', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 只在第一个图显示图例，避免乱
        if i == 0:
            ax.legend(loc='upper right', fontsize='small')

    # 隐藏多余的子图
    for i in range(num_joints, len(axes)):
        axes[i].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # 留出标题空间

    plt.show()


def plot_tactile_digital(csv_file="isaacgymenvs/sim_joint_data.csv"):
    try:
        df = pd.read_csv(csv_file)
        print(f"成功读取数据: {len(df)} 步")
    except FileNotFoundError:
        print("未找到 csv 文件")
        return

    # 1. 筛选触觉列
    tactile_cols = [c for c in df.columns if 'fsr' in c or 'tactile' in c]
    
    if not tactile_cols:
        print("未检测到触觉数据列。")
        return

    num_sensors = len(tactile_cols)
    steps = df.index
    
    # 设置画布大小，高度随传感器数量自动调整
    fig, ax = plt.subplots(figsize=(15, 0.6 * num_sensors + 2))
    
    # 定义垂直间距
    y_offset_step = 1.3 
    
    # 定义不同手指的颜色组 (为了美观，循环使用颜色)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    print("正在绘制逻辑波形图...")

    for i, col_name in enumerate(tactile_cols):
        # 基础高度 (Baseline)
        base_y = i * y_offset_step
        
        # 获取 0/1 数据
        signal = df[col_name].values
        
        # 将 0/1 信号叠加到基础高度上
        # signal * 0.9 是为了让波形高度不要完全填满间距，留点空隙
        plot_data = base_y + signal * 0.9
        
        # === 关键绘图逻辑 ===
        # drawstyle='steps-pre' 画出直角方波
        color = colors[i % 10]
        ax.plot(steps, plot_data, 
                color=color, 
                linewidth=1.5, 
                drawstyle='steps-pre')
        
        # 填充"激活"区域 (模拟高电平)
        # fill_between 也要使用 step='pre' 才能和线条对齐
        ax.fill_between(steps, base_y, plot_data, 
                        step='pre', 
                        color=color, 
                        alpha=0.3) # alpha 是透明度
        
        # 画一条灰色的基准线
        ax.axhline(y=base_y, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

    # === 设置坐标轴 ===
    # 设置 Y 轴刻度标签为传感器名称
    tick_locs = [i * y_offset_step + 0.45 for i in range(num_sensors)]
    ax.set_yticks(tick_locs)
    ax.set_yticklabels(tactile_cols, fontsize=10)
    
    ax.set_xlabel('Simulation Steps')
    ax.set_title('Tactile Sensor Activation (0/1 Logic View)', fontsize=14)
    ax.grid(True, axis='x', alpha=0.3) # 只显示X轴网格方便看时间对齐
    
    # 去掉顶部和右侧的边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_tactile_digital()
    plot_simulation_data()