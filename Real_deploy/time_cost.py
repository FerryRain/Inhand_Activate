"""
@FileName：time_cost.py
@Description：
@Author：Ferry
@Time：2026 1/13/26 4:24 PM
@Copyright：©2024-2026 ShanghaiTech University-RIMLAB
"""

import os
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
            # main stages
            "stage/tracking",
            "stage/reconstruct",
            "stage/active",

            # hand-only breakdown (get_axis calls)
            "hand/tracking",
            "hand/reconstruct",
            "hand/active",

            # optional breakdown for active
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

    starts_axis = "z"
    tracker = Tracker(text_prompt="A green object")
    recon = Reconstructor()
    est = GPISNBVv2()

    WINDOW_NAME = "Contact Data_left"
    controller = HardwarePlayer(config)

    # --- deployment (not counted into tracking/re/active; if you want, you can time them too) ---
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

    # count how many times ac(active) is used
    ac_count = 0

    try:
        while True:
            if tracker.init_done:
                # -------- tracking stage --------
                with timed(stats, "stage/tracking"):
                    key = tracker.tracking()

                if key == "quit":
                    cv2.destroyAllWindows()
                    controller.stop()
                    break

                if key == "reconstruct":
                    # -------- reconstruct stage --------
                    with timed(stats, "stage/reconstruct"):
                        with timed(stats, "hand/reconstruct"):
                            controller.get_axis()

                        pcd = recon.reconstruct()
                        recon.show()

                if key == "active":
                    ac_count += 1

                    # -------- active stage --------
                    with timed(stats, "stage/active"):
                        # (1) hand axis query
                        # with timed(stats, "hand/active"):
                        #     controller.get_axis()
                        controller.get_axis()
                        # # (2) reconstruct
                        with timed(stats, "active/reconstruct"):
                            pcd = recon.reconstruct()

                        # (3) NBV estimate
                        with timed(stats, "active/nbv_estimate"):
                            nbv = est.estimate(pcd, seed=0, verbose=True)

                        # (4) NBV viz
                        # with timed(stats, "active/nbv_viz"):
                        est.viz()

                        # (5) choose world axis
                        with timed(stats, "active/pick_axis"):
                            axis_idx, rvec_W, aW, vW, T_WO, T_WC = pick_world_axis_and_rvec(
                                tracker.T, tracker.init_pose, nbv["best_dir"]
                            )

                        print(f"Active success! Turn to rotating along {axis_name[axis_idx]} from {rotating_now}")
                        rotating_now = axis_name[axis_idx]

                        # (6) hand set axis (actual command)
                        with timed(stats, "active/hand_set_axis"):
                            controller.get_axis(rotating_now)

            else:
                key = tracker.init_tracker()
                if key == "quit":
                    cv2.destroyAllWindows()
                    controller.stop()
                    break
                if key == "start":
                    # this belongs to "tracking hand time" for your request
                    with timed(stats, "hand/tracking"):
                        controller.get_axis(starts_axis)
                    rotating_now = starts_axis

    finally:
        # -------- print + save summary --------
        print("\n================ Timing Summary ================")
        for l in stats.summary_lines():
            print(l)
        print(f"\nac(active) used count = {ac_count}")

        # save into current (Hydra) working directory
        out_json = os.path.join(os.getcwd(), "timing_stats.json")
        dump_stats(stats, out_json)
        print(f"Saved timing stats to: {out_json}")
        print("================================================\n")


if __name__ == '__main__':
    main()
