"""
@FileName：main_explore.py
@Description：
@Author：Ferry
@Time：2026 1/9/26 3:07 PM
@Copyright：©2024-2026 ShanghaiTech University-RIMLAB
"""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import hydra
from omegaconf import DictConfig
import cv2

from reconstruction.Reconstructor import Reconstructor
# from Tracking.tracker import Tracker
# from Tracking.tracker_fast_sam import Tracker
from Tracking.sam3_tracking import Tracker
from Active.NBV_gpis import GPISNBVv2
from LeapHand_rotation.isaacgymenvs.hand_controller import HardwarePlayer

from Active.motion_planner import pick_world_axis_and_rvec




@hydra.main(config_name='config', config_path='../LeapHand_rotation/isaacgymenvs/cfg')
def main(config: DictConfig):
    tracker = Tracker(text_prompt="An green object")
    # tracker = Tracker(text_prompt="A yellow object")
    recon = Reconstructor()
    est = GPISNBVv2()

    WINDOW_NAME = "Contact Data_left"
    controller = HardwarePlayer(config)
    controller.restore_all_models()
    controller.start_deployment()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 400, 400)
    controller.get_axis()

    axis_name = ["x", "y", "z"]
    pcd = None
    rotating_now = "stop"
    while True:
        if tracker.init_done:
            key = tracker.tracking()
            # print(tracker.T)
            if key == "quit":
                cv2.destroyAllWindows()
                controller.stop()
                break
            if key == "reconstruct":
                controller.get_axis()
                pcd = recon.reconstruct()
                recon.show()
            if key == "active":
                controller.get_axis()
                pcd = recon.reconstruct()
                # recon.show()
                nbv = est.estimate(pcd, seed=0, verbose=True)
                est.viz()
                axis_idx, rvec_W, aW, vW, T_WO, T_WC = pick_world_axis_and_rvec(tracker.T, tracker.init_pose, nbv["best_dir"])
                print(f"Active success! Turn to rotating along {axis_name[axis_idx]} from {rotating_now}")
                rotating_now = axis_name[axis_idx]
                controller.get_axis(rotating_now)

        else:
            key = tracker.init_tracker()
            if key == "quit":
                cv2.destroyAllWindows()
                controller.stop()
                break
            if key == "start":
                controller.get_axis("x")
                rotating_now = "x"



if __name__ == '__main__':
    main()

    # tracker = Tracker(text_prompt="An yellow cylinder with stickers attached")
    # while True:
    #     if tracker.init_done:
    #         key = tracker.tracking()
    #     else:
    #         key = tracker.init_tracker()
    #         if key == "quit":
    #             cv2.destroyAllWindows()
    #             break
