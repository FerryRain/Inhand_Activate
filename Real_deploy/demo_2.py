"""
@FileName：time_cost.py
@Description：
@Author：Ferry
@Time：2026 1/13/26 4:24 PM
@Copyright：©2024-2026 ShanghaiTech University-RIMLAB
"""

import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import sys
sys.path.append("/home/ferry/data/Code2/Research/Inhand_Activate")

# 可选：降低多线程库的抖动/抢占（如果你发现 GPIS 反而变慢，就注释掉）
# os.environ.setdefault("OMP_NUM_THREADS", "1")
# os.environ.setdefault("MKL_NUM_THREADS", "1")
# os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
# os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import time
import json
from dataclasses import dataclass, field
from collections import defaultdict
from contextlib import contextmanager

import hydra
from omegaconf import DictConfig
import cv2

from reconstruction.Reconstructor import Reconstructor
from Tracking.tracking_offline_sam3 import OfflineDiskRecorderTracker
from Active.NBV_gpis import GPISNBVv2
from LeapHand_rotation.isaacgymenvs.hand_controller import HardwarePlayer
from Active.motion_planner import pick_world_axis_and_rvec


# ----------------------------- Timing Utils -----------------------------
@dataclass
class TimingStats:
    sums: dict = field(default_factory=lambda: defaultdict(float))
    counts: dict = field(default_factory=lambda: defaultdict(int))

    def add(self, name: str, dt: float):
        self.sums[name] += float(dt)
        self.counts[name] += 1

    def summary_lines(self):
        def line(name: str):
            total = self.sums.get(name, 0.0)
            n = self.counts.get(name, 0)
            avg = total / n if n else 0.0
            return f"{name:28s} total={total:9.3f}s  n={n:6d}  avg={avg:8.4f}s"

        order = [
            "stage/tracking",
            "stage/reconstruct",
            "stage/active",
            "stage/Manip",              # <- 修复：这里必须有逗号

            "hand/tracking",
            "hand/reconstruct",
            "hand/active",

            "active/reconstruct",
            "active/nbv_estimate",
            "active/nbv_viz",
            "active/pick_axis",
            "active/hand_set_axis",
            "active/tracking",
        ]
        return [line(k) for k in order if (k in self.sums or k in self.counts)]


@contextmanager
def timed(stats: TimingStats, name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        stats.add(name, time.perf_counter() - t0)


def dump_stats(stats: TimingStats, out_path: str):
    payload = {"sums_sec": dict(stats.sums), "counts": dict(stats.counts)}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def sleep_until(t_target: float, max_sleep: float = 0.005):
    """
    以较小粒度 sleep，避免长 sleep 造成调度不稳定。
    """
    while True:
        now = time.perf_counter()
        dt = t_target - now
        if dt <= 0:
            return
        time.sleep(min(max_sleep, dt))


# ----------------------------- Main -----------------------------
@hydra.main(config_name='config', config_path='../LeapHand_rotation/isaacgymenvs/cfg')
def main(config: DictConfig):
    out_dir = "/demo/demo/Cross/001"
    starts_axis = "z"

    stats = TimingStats()

    # >>> Config
    ACTIVE_PERIOD_SEC = 6.0
    MANIP_BUDGET_SEC = 30.0

    SAVE_FPS = 20.0
    SAVE_DT = 1.0 / SAVE_FPS
    RESYNC_LAG_SEC = 0.2  # 落后太多就重置节拍，避免“追帧”导致更卡

    VIZ_EVERY_N_ACTIVE = 1  # 如果你觉得 est.viz() 卡，把它改成 2/3/5 做降频


    # OpenCV：减少线程带来的抖动/抢占（通常会更稳）
    try:
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass

    # tracker：初始化阶段需要 UI，进入主循环后尽量不要 GUI
    tracker = OfflineDiskRecorderTracker(text_prompt="A green object", show_ui=True, root_dir=out_dir, k4a_color_res="720P",)

    recon = Reconstructor()
    est = GPISNBVv2()

    controller = HardwarePlayer(config, com="/dev/ttyUSB1")
    controller.restore_all_models()
    controller.start_deployment()

    # ---------------- init stage (手动录制/画 gate) ----------------
    WINDOW_NAME = "Contact Data_left"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 400, 400)

    with timed(stats, "hand/tracking"):
        controller.get_axis()

    axis_name = ["z", "x", "z", "z","x"]
    pcd = None
    rotating_now = "stop"
    ac_count = 0
    init_done = False

    next_active_t = 0.0
    next_save_t = 0.0

    try:
        while True:
            if not init_done:
                color, _ = tracker.camera.get_k4a_frame(require_aligned_depth=True)
                disp = color.copy()
                cv2.putText(
                    disp,
                    f"disk_dir={os.path.basename(tracker.root_dir)}  idx_next={tracker.idx_next}  init_done={tracker.init_done}  last_tracked={tracker.last_tracked_idx}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow(tracker.win, disp)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    cv2.destroyAllWindows()
                    controller.stop()
                    break
                elif key == ord("r"):
                    # 手动单帧录制
                    with timed(stats, "stage/Manip"):
                        tracker.record_frame()
                elif key == ord("g"):
                    # 画 gate 并进入主循环
                    tracker.draw_gate(idx=0)
                    init_done = True

                    # 进入主循环后，尽量关闭 tracker UI（若类里支持）
                    if hasattr(tracker, "show_ui"):
                        try:
                            tracker.show_ui = False
                        except Exception:
                            pass

                    now = time.perf_counter()
                    next_active_t = now + ACTIVE_PERIOD_SEC
                    next_save_t = now  # 立即开始按 30FPS 保存

                    controller.get_axis(starts_axis)
                    cv2.destroyAllWindows()

            # ---------------- main loop: fixed-rate saving (30 FPS) ----------------
            if init_done:
                with timed(stats, "stage/Manip"):
                    # 1) 固定节拍保存（30 FPS）
                    now = time.perf_counter()

                    if now < next_save_t:
                        sleep_until(next_save_t)
                        now = time.perf_counter()

                    lag = now - next_save_t
                    if lag > RESYNC_LAG_SEC:
                        next_save_t = now


                    tracker.record_frame()

                    next_save_t += SAVE_DT

                    # 2) 到点触发 active（注意：active 会阻塞主循环，这里是“可接受的最小改法”）
                    now = time.perf_counter()
                manip_used = stats.sums.get("stage/Manip", 0.0)
                if manip_used >= MANIP_BUDGET_SEC:
                    print(f"\n[STOP] stage/Manip reached {MANIP_BUDGET_SEC:.3f}s (budget={MANIP_BUDGET_SEC:.1f}s). Exiting.")
                    controller.stop()
                    tracker.track()
                    break

                if now >= next_active_t:
                    ac_count += 1
                #
                    controller.get_axis()
                    controller.get_axis(axis_name[ac_count])
                    next_active_t = time.perf_counter() + ACTIVE_PERIOD_SEC
                #
                #     with timed(stats, "active/tracking"):
                #         tracker.track()
                #
                #     with timed(stats, "stage/active"):
                #         with timed(stats, "active/reconstruct"):
                #             pcd = recon.reconstruct()
                #
                #         with timed(stats, "active/nbv_estimate"):
                #             nbv = est.estimate(pcd, seed=0, verbose=True)
                #
                #         # est.viz() 往往很卡：建议降频或关掉
                #         # if (ac_count % VIZ_EVERY_N_ACTIVE) == 0:
                #         #     with timed(stats, "active/nbv_viz"):
                #         est.viz()
                #
                #         with timed(stats, "active/pick_axis"):
                #             axis_idx, rvec_W, aW, vW, T_WO, T_WC = pick_world_axis_and_rvec(
                #                 tracker.last_T, tracker.init_pose, nbv["best_dir"]
                #             )
                #
                #         print(f"[Active@{ac_count}] Turn to rotating along {axis_name[axis_idx]} from {rotating_now}")
                #         rotating_now = axis_name[axis_idx]
                #
                #         with timed(stats, "active/hand_set_axis"):
                #             controller.get_axis(rotating_now)
                #
                #     next_active_t = time.perf_counter() + ACTIVE_PERIOD_SEC

    except KeyboardInterrupt:
        print("\n[KeyboardInterrupt] Exiting...")

    finally:
        print("\n================ Timing Summary ================")
        for l in stats.summary_lines():
            print(l)
        print(f"\nac(active) used count = {ac_count}")

        out_json = os.path.join(os.getcwd(), "timing_stats.json")
        dump_stats(stats, out_json)
        print(f"Saved timing stats to: {out_json}")
        print("================================================\n")


if __name__ == '__main__':
    main()
