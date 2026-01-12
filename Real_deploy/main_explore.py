"""
@FileName：main_explore.py
@Description：
@Author：Ferry
@Time：2026 1/9/26 3:07 PM
@Copyright：©2024-2026 ShanghaiTech University-RIMLAB
"""
from reconstruction.Reconstructor import Reconstructor
from Tracking.tracker import Tracker
# from Tracking.tracker_fast_sam import Tracker
# from Tracking.sam3_tracking import Tracker
from Active.NBV_gpis import GPISNBVv2

if __name__ == '__main__':
    tracker = Tracker()
    recon = Reconstructor()
    est = GPISNBVv2()

    pcd = None
    while True:
        if tracker.init_done:
            key = tracker.tracking()
            # print(tracker.T)
            if key == "quit":
                break
            if key == "reconstruct":
                pcd = recon.reconstruct()
                recon.show()
            if key == "active":
                pcd = recon.reconstruct()
                recon.show()
                nbv = est.estimate(pcd, seed=0, verbose=True)
                est.viz()
        else:
            quit = tracker.init_tracker()
            if quit == "quit":
                break