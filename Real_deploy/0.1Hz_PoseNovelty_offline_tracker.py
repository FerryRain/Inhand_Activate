"""
@FileName：0.1Hz_PoseNovelty_offline_tracker.py
@Description：Offline-tracking real deployment with Pose-Novelty active planning.
@Author：Ferry
@Time：2026 8/8
@Copyright：©2024-2026 ShanghaiTech University-RIMLAB

This program intentionally mirrors ``0.1Hz_Actite_offline_tracker.py``.  The
camera recording, SAM3 + BundleTrack data flow, reconstruction, timing budget,
and low-level hand controller are unchanged; only the active axis-selection
module is replaced by Pose-Novelty.
"""

import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import sys

PROJECT_ROOT = "/home/ferry/data/Code2/Research/Inhand_Activate"
sys.path.append(PROJECT_ROOT)

import json
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field

import cv2
import hydra
from omegaconf import DictConfig

from Active.pose_novelty import PoseNoveltyAxisPlanner, load_calibrated_angles
from LeapHand_rotation.isaacgymenvs.hand_controller import HardwarePlayer
from Tracking.tracking_offline_sam3 import OfflineDiskRecorderTracker
from reconstruction.Reconstructor import Reconstructor


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
            "stage/Manip",
            "hand/tracking",
            "hand/reconstruct",
            "hand/active",
            "active/reconstruct",
            "active/pose_novelty",
            "active/hand_set_axis",
            "active/tracking",
        ]
        return [line(key) for key in order if (key in self.sums or key in self.counts)]


@contextmanager
def timed(stats: TimingStats, name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        stats.add(name, time.perf_counter() - start)


def dump_stats(stats: TimingStats, out_path: str):
    payload = {"sums_sec": dict(stats.sums), "counts": dict(stats.counts)}
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def append_jsonl(out_path: str, payload: dict):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def sleep_until(target_time: float, max_sleep: float = 0.005):
    """Sleep at a small granularity to avoid scheduler jitter."""

    while True:
        now = time.perf_counter()
        remaining = target_time - now
        if remaining <= 0:
            return
        time.sleep(min(max_sleep, remaining))


# ----------------------------- Main -----------------------------
@hydra.main(config_name="config", config_path="../LeapHand_rotation/isaacgymenvs/cfg")
def main(config: DictConfig):
    # Keep Pose-Novelty raw data separate from Ray-GPIS runs.
    out_dir = os.path.join(
        PROJECT_ROOT,
        "Real_deploy/results/offline_tracking/pose_novelty/cube_obj_02/004",
    )
    stats = TimingStats()

    # Same acquisition/control protocol as 0.1Hz_Actite_offline_tracker.py.
    ACTIVE_PERIOD_SEC = 6.0
    MANIP_BUDGET_SEC = 50.0
    SAVE_FPS = 20.0
    SAVE_DT = 1.0 / SAVE_FPS
    RESYNC_LAG_SEC = 0.2
    starts_axis = "z"

    calibration_path = os.path.join(
        PROJECT_ROOT,
        "rebuttal/assets/action_calibration.json",
    )
    pose_state_path = os.path.join(out_dir, "pose_novelty_state.json")
    decision_log_path = os.path.join(out_dir, "pose_novelty_decisions.jsonl")

    try:
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass

    tracker = OfflineDiskRecorderTracker(
        text_prompt="A green object",
        show_ui=True,
        root_dir=out_dir,
        k4a_color_res="720P",
    )
    recon = Reconstructor()
    pose_planner = PoseNoveltyAxisPlanner(load_calibrated_angles(calibration_path))
    if pose_planner.load_state(pose_state_path):
        print(
            f"[PoseNovelty] resumed {pose_planner.history_size} pose records "
            f"from {pose_state_path}"
        )

    controller = HardwarePlayer(config, com="/dev/ttyUSB1")
    controller.restore_all_models()
    controller.start_deployment()

    # ---------------- init stage (manual record / gate) ----------------
    window_name = "Contact Data_left"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 400, 400)

    with timed(stats, "hand/tracking"):
        controller.get_axis()

    pcd = None
    rotating_now = "stop"
    active_count = 0
    init_done = False
    next_active_time = 0.0
    next_save_time = 0.0

    try:
        while True:
            if not init_done:
                color, _ = tracker.camera.get_k4a_frame(require_aligned_depth=True)
                display = color.copy()
                cv2.putText(
                    display,
                    (
                        f"disk_dir={os.path.basename(tracker.root_dir)}  "
                        f"idx_next={tracker.idx_next}  init_done={tracker.init_done}  "
                        f"last_tracked={tracker.last_tracked_idx}"
                    ),
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow(tracker.win, display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    cv2.destroyAllWindows()
                    controller.stop()
                    break
                if key == ord("r"):
                    with timed(stats, "stage/Manip"):
                        tracker.record_frame()
                elif key == ord("g"):
                    tracker.draw_gate(idx=0)
                    init_done = True

                    if hasattr(tracker, "show_ui"):
                        try:
                            tracker.show_ui = False
                        except Exception:
                            pass

                    now = time.perf_counter()
                    next_active_time = now + ACTIVE_PERIOD_SEC
                    next_save_time = now
                    controller.get_axis(starts_axis)
                    rotating_now = starts_axis
                    cv2.destroyAllWindows()

            # ---------------- main loop: fixed-rate saving (20 FPS) ----------------
            if init_done:
                with timed(stats, "stage/Manip"):
                    now = time.perf_counter()
                    if now < next_save_time:
                        sleep_until(next_save_time)
                        now = time.perf_counter()

                    lag = now - next_save_time
                    if lag > RESYNC_LAG_SEC:
                        next_save_time = now

                    tracker.record_frame()
                    next_save_time += SAVE_DT
                    now = time.perf_counter()

                manipulation_used = stats.sums.get("stage/Manip", 0.0)
                if manipulation_used >= MANIP_BUDGET_SEC:
                    print(
                        f"\n[STOP] stage/Manip reached {MANIP_BUDGET_SEC:.3f}s "
                        f"(budget={MANIP_BUDGET_SEC:.1f}s). Exiting."
                    )
                    controller.stop()
                    tracker.track()
                    break

                if now >= next_active_time:
                    active_count += 1

                    # No-argument get_axis() stops the current rotation policy.
                    controller.get_axis()

                    with timed(stats, "active/tracking"):
                        tracker.track()

                    with timed(stats, "stage/active"):
                        # Keep the common reconstruction path for identical output
                        # data and downstream evaluation; Pose-Novelty never reads pcd.
                        with timed(stats, "active/reconstruct"):
                            pcd = recon.reconstruct()

                        with timed(stats, "active/pose_novelty"):
                            if tracker.last_T is None:
                                raise RuntimeError(
                                    "Offline tracking produced no current pose for Pose-Novelty"
                                )

                            initial_pose = getattr(tracker, "init_pose", None)
                            if initial_pose is not None:
                                pose_planner.add_pose(
                                    initial_pose,
                                    pose_id="tracker/initial_pose",
                                )

                            keyframe_root = os.path.join(recon.debug_dir, "keyframes")
                            new_keyframes = pose_planner.update_from_keyframes(keyframe_root)

                            decision_pose_id = f"decision/{int(tracker.last_tracked_idx):06d}"
                            pose_planner.add_pose(
                                tracker.last_T,
                                pose_id=decision_pose_id,
                            )
                            reference_pose = (
                                initial_pose
                                if initial_pose is not None
                                else pose_planner.reference_pose
                            )
                            selected_axis, decision = pose_planner.select_axis(
                                tracker.last_T,
                                initial_pose_co=reference_pose,
                            )

                        decision.update(
                            {
                                "active_step": int(active_count),
                                "last_tracked_idx": int(tracker.last_tracked_idx),
                                "new_valid_keyframes": int(new_keyframes),
                                "rotating_from": rotating_now,
                                "wall_time_unix": float(time.time()),
                                "uses_reconstruction": False,
                            }
                        )
                        pose_planner.save_state(pose_state_path)
                        append_jsonl(decision_log_path, decision)

                        score_text = ", ".join(
                            f"{axis}={decision['scores_deg'][axis]:.2f}deg"
                            for axis in ("x", "y", "z")
                        )
                        print(
                            f"[PoseNovelty@{active_count}] scores: {score_text}; "
                            f"history={decision['history_size']}, "
                            f"new_keyframes={new_keyframes}"
                        )
                        print(
                            f"[Active@{active_count}] Turn to rotating along "
                            f"{selected_axis} from {rotating_now}"
                        )
                        rotating_now = selected_axis

                        with timed(stats, "active/hand_set_axis"):
                            controller.get_axis(rotating_now)

                    next_active_time = time.perf_counter() + ACTIVE_PERIOD_SEC

    except KeyboardInterrupt:
        print("\n[KeyboardInterrupt] Exiting...")

    finally:
        print("\n================ Timing Summary ================")
        for line in stats.summary_lines():
            print(line)
        print(f"\nac(active) used count = {active_count}")

        out_json = os.path.join(os.getcwd(), "timing_stats_pose_novelty.json")
        dump_stats(stats, out_json)
        print(f"Saved timing stats to: {out_json}")
        print(f"Saved Pose-Novelty decisions to: {decision_log_path}")
        print("================================================\n")


if __name__ == "__main__":
    main()
