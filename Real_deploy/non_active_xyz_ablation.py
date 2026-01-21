"""
@FileName：time_cost.py
@Description：
@Author：Ferry
@Time：2026 1/13/26 4:24 PM
@Copyright：©2024-2026 ShanghaiTech University-RIMLAB
"""

import os
from idlelib.configdialog import changes

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import time
import json
from dataclasses import dataclass, field
from collections import defaultdict
from contextlib import contextmanager

import hydra
from omegaconf import DictConfig
import cv2

from reconstruction.Reconstructor import Reconstructor
from Tracking.sam3_tracking import Tracker
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

            "hand/tracking",
            "hand/reconstruct",
            "hand/active",

            "active/reconstruct",
            "active/nbv_estimate",
            "active/nbv_viz",
            "active/pick_axis",
            "active/hand_set_axis",
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
    payload = {
        "sums_sec": dict(stats.sums),
        "counts": dict(stats.counts),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


# ----------------------------- Main -----------------------------
@hydra.main(config_name='config', config_path='../LeapHand_rotation/isaacgymenvs/cfg')
def main(config: DictConfig):
    stats = TimingStats()

    # >>> Config
    ACTIVE_PERIOD_SEC = 20
    TRACKING_BUDGET_SEC = 60.0  # stop when accumulated stage/tracking >= 60s

    next_active_t = None  # schedule for periodic active (wall-clock)
    next_active_t_2 = None
    starts_axis = "z"
    tracker = Tracker(text_prompt="A purple object",show_tracker=False)
    recon = Reconstructor()
    est = GPISNBVv2()

    WINDOW_NAME = "Contact Data_left"
    controller = HardwarePlayer(config, com="/dev/ttyUSB0")

    controller.restore_all_models()
    controller.start_deployment()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 400, 400)

    # initial axis query/set
    with timed(stats, "hand/tracking"):
        controller.get_axis()

    axis_name = ["x", "y", "z"]
    pcd = None
    rotating_now = "stop"

    ac_count = 0  # active times
    change_count = 0
    try:
        while True:
            if tracker.init_done:
                # -------- tracking stage --------
                with timed(stats, "stage/tracking"):
                    key = tracker.tracking()

                # quit immediately
                if key == "quit":
                    cv2.destroyAllWindows()
                    controller.stop()
                    break

                # >>> NEW: stop when tracking time reaches budget
                tracking_used = stats.sums.get("stage/tracking", 0.0)
                if tracking_used >= TRACKING_BUDGET_SEC:
                    print(f"\n[STOP] stage/tracking reached {tracking_used:.3f}s "
                          f"(budget={TRACKING_BUDGET_SEC:.1f}s). Exiting.")
                    cv2.destroyAllWindows()
                    controller.stop()
                    break

                # optional: keep manual reconstruct
                if key == "reconstruct":
                    with timed(stats, "stage/reconstruct"):
                        with timed(stats, "hand/reconstruct"):
                            controller.get_axis()
                        pcd = recon.reconstruct()
                        recon.show()

                # -------- periodic active trigger (every 10s wall-clock) --------
                now = time.perf_counter()
                if next_active_t is None:
                    next_active_t_2 = now + 2 * ACTIVE_PERIOD_SEC
                    next_active_t = now + ACTIVE_PERIOD_SEC
                    change_count +=1
                    pass

                if now >= next_active_t and now < next_active_t_2:
                    controller.get_axis("")
                    # print("change")
                    controller.get_axis("x")
                elif now >= next_active_t_2 and change_count !=2:
                    controller.get_axis("")
                    print("change")
                    controller.get_axis("y")
                    change_count = 2


            else:
                key = tracker.init_tracker()
                if key == "quit":
                    cv2.destroyAllWindows()
                    controller.stop()
                    break
                if key == "start":
                    with timed(stats, "hand/tracking"):
                        controller.get_axis(starts_axis)
                    rotating_now = starts_axis

                    # start periodic active schedule after tracking begins
                    # next_active_t = time.perf_counter() + ACTIVE_PERIOD_SEC
                    cv2.destroyAllWindows()

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
