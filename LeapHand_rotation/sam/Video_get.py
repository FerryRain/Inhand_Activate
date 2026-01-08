import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
import json

# --- 配置参数 ---
WIDTH = 640
HEIGHT = 480
FRAMERATE = 30
base_folder = 'realsense_data_auto'  # 主文件夹名称

# 【关键设置】自动抓取间隔 (秒)
# 0.1 = 每秒10张, 0.5 = 每秒2张, 1.0 = 每秒1张
CAPTURE_INTERVAL = 0.1

# 抗频闪: 1=50Hz, 2=60Hz
POWER_LINE_FREQ = 1

# --- 创建输出目录 ---
run_timestamp = int(time.time())
OUTPUT_DIR = os.path.join(base_folder, f"run_{run_timestamp}")

dirs = {
    'root': OUTPUT_DIR,
    'rgb': os.path.join(OUTPUT_DIR, 'rgb'),
    'depth': os.path.join(OUTPUT_DIR, 'depth'),  # 16位原始深度
    'vis': os.path.join(OUTPUT_DIR, 'depth_vis') # 可视化深度图
}

for k, path in dirs.items():
    if not os.path.exists(path):
        os.makedirs(path)

# --- 初始化 RealSense ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FRAMERATE)
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FRAMERATE)

print("正在启动相机...")
profile = pipeline.start(config)

# 设置抗频闪
color_sensor = profile.get_device().first_color_sensor()
if color_sensor.supports(rs.option.power_line_frequency):
    color_sensor.set_option(rs.option.power_line_frequency, POWER_LINE_FREQ)

# --- 1. 导出相机内参 ---
print("正在导出内参...")
align = rs.align(rs.stream.color) # 深度对齐到彩色
color_stream = profile.get_stream(rs.stream.color)
intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

intrinsics_dict = {
    "width": intrinsics.width,
    "height": intrinsics.height,
    "fx": intrinsics.fx,
    "fy": intrinsics.fy,
    "ppx": intrinsics.ppx,
    "ppy": intrinsics.ppy,
    "model": str(intrinsics.model),
    "coeffs": intrinsics.coeffs
}

with open(os.path.join(OUTPUT_DIR, 'intrinsics.json'), 'w') as f:
    json.dump(intrinsics_dict, f, indent=4)
print("内参已保存。")

# --- 视频录制 (可选，如果不需要可以注释掉) ---
# 注意：即使不录像，下面的预览窗口依然会显示
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
color_writer = cv2.VideoWriter(os.path.join(OUTPUT_DIR, 'preview_color.avi'), fourcc, FRAMERATE, (WIDTH, HEIGHT))

# --- 变量初始化 ---
frame_count = 0
last_save_time = time.time()
print("-" * 50)
print(f"开始自动采集！")
print(f"采集频率: 每 {CAPTURE_INTERVAL} 秒一张")
print(f"数据保存至: {OUTPUT_DIR}")
print(f"按 'q' 键停止程序")
print("-" * 50)

try:
    while True:
        # 1. 获取并对齐帧
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # 2. 转换数据
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # 深度图伪彩 (仅用于显示)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # 3. 自动保存逻辑
        current_time = time.time()
        # 如果当前时间与上次保存时间的差值大于设定的间隔
        if current_time - last_save_time >= CAPTURE_INTERVAL:
            
            # 构造文件名 (使用帧号)
            fname = f"{frame_count:06d}"
            
            # 保存彩色图 (PNG)
            cv2.imwrite(os.path.join(dirs['rgb'], f"{fname}.jpg"), color_image)
            
            # 保存深度图 (16-bit PNG, 包含真实距离信息)
            cv2.imwrite(os.path.join(dirs['depth'], f"{fname}.png"), depth_image)
            
            # (可选) 保存可视化深度图
            cv2.imwrite(os.path.join(dirs['vis'], f"{fname}.jpg"), depth_colormap)
            
            print(f"[已保存] 帧: {frame_count}")
            
            # 更新状态
            last_save_time = current_time
            frame_count += 1

        # 4. 显示与录制视频流
        color_writer.write(color_image)
        
        # 拼接显示
        images = np.hstack((color_image, depth_colormap))
        cv2.putText(images, f"Saved: {frame_count}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Auto Capture (Press q to quit)', images)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    color_writer.release()
    pipeline.stop()
    cv2.destroyAllWindows()
    print(f"采集结束。共保存 {frame_count} 组图片。")