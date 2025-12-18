"""
@FileName：depth_check.py
@Description：
@Author：Ferry
@Time：2025 12/17/25 4:28 PM
@Copyright：©2024-2025 ShanghaiTech University-RIMLAB
"""
import cv2, numpy as np
d = cv2.imread("/home/ferry/data/Code2/Research/PoseEstimation/BundleTrack/YCBInEOAT/inhand_object2/depth/00013_1765959166369.png", cv2.IMREAD_UNCHANGED)
print(d.dtype, d.min(), d.max(), np.median(d[d>0]))