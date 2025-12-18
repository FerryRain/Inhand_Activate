"""
@FileName：intrinsics_transform.py
@Description：
@Author：Ferry
@Time：2025 12/17/25 3:27 PM
@Copyright：©2024-2025 ShanghaiTech University-RIMLAB
"""
import json

json_path = "/home/ferry/data/Code2/Research/PoseEstimation/BundleTrack/YCBInEOAT/inhand_object2/intrinsics.json"
out_path  = "/home/ferry/data/Code2/Research/PoseEstimation/BundleTrack/YCBInEOAT/inhand_object2/cam_K.txt"

with open(json_path, "r") as f:
    intr = json.load(f)

fx, fy = intr["fx"], intr["fy"]
ppx, ppy = intr["ppx"], intr["ppy"]

K = [
    [fx, 0.0, ppx],
    [0.0, fy, ppy],
    [0.0, 0.0, 1.0],
]

with open(out_path, "w") as f:
    for row in K:
        f.write(" ".join(f"{v:.18e}" for v in row) + "\n")

print("Saved:", out_path)
