"""
@FileName：main_explore.py
@Description：
@Author：Ferry
@Time：2026 1/9/26 3:07 PM
@Copyright：©2024-2026 ShanghaiTech University-RIMLAB
"""
from reconstruction.Reconstructor import Reconstructor
from Real_deploy.utils.tracker import Tracker

if __name__ == '__main__':
    tracker = Tracker()
    recon = Reconstructor()

    while True:
        if tracker.init_done:
            key = tracker.tracking()
            # print(tracker.T)
            if key == "quit":
                break
            if key == "reconstruct":
                pcd = recon.reconstruct()
                recon.show()
        else:
            quit = tracker.init_tracker()
            if quit == "quit":
                break


    pcd = recon.reconstruct()
    recon.show()
    recon.save_color("./color.ply")
    recon.save_xyz_ascii("./xyz_ascii.ply")

    # recon.reconstruct(save_color="./color.ply", save_xyz_ascii="./xyz_ascii.ply")