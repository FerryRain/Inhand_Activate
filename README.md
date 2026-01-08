# Inhand_Activate
## 6D Tracking (BundleTrack)

This project uses **BundleTrack** for 6D object tracking. We support two modes:

* **Offline (folder/dataset testing)**: run tracking on a local dataset folder
* **Online (real-time testing)**: stream frames from a camera to a running BundleTrack server

Both modes share the same setup: you must enter **BundleTrack** and run **two terminals**:

1. an **lf-net** feature server
2. the **BundleTrack** docker environment

---

### 0) Enter BundleTrack

All commands below assume you are in the BundleTrack repository:

```bash
cd [PATH_TO_BUNDLETRACK]
```

---

## A. Shared Setup (Required)

### A1) Terminal 1 — Start lf-net feature server

```bash
bash lf-net-release/docker/run_container.sh
cd [PATH_TO_BUNDLETRACK]
cd lf-net-release && python run_server.py
```
> Open new bash of the docker

> docker exec -it bundletrack /bin/bash
> 
>  cd home/ferry/data/Code2/Research/Inhand_Activate/BundleTrack/
> 

> docker exec -it lfnet /bin/bash
> 
>  cd home/ferry/data/Code2/Research/Inhand_Activate/BundleTrack

> Keep this terminal running. BundleTrack uses it for feature detection/matching.

---

### A2) Terminal 2 — Start BundleTrack docker

```bash
bash docker/run_container.sh
cd [PATH_TO_BUNDLETRACK]
```

```bash
docker exec -it  bundletrack bash
```

Then follow either **Offline** or **Online** mode below.

---

## B. Offline Test (Folder / Dataset)

Inside the **BundleTrack docker terminal (Terminal 2)** run:

```bash
python scripts/run_mydataset.py
```

Outputs will be written under `BundleTrack/results/...` (exact subfolder depends on your dataset/script configuration).

---

## C. Online Test (Real-time Tracking)

Real-time tracking consists of:

1. launching the **BundleTrack server** inside docker
2. running the **local sender** script that captures frames and sends them to BundleTrack

---

### C1) BundleTrack docker — Start real-time server

Inside the **BundleTrack docker terminal (Terminal 2)**:

```bash
python scripts/real_time.py
```

---

### C2) Camera intrinsics (Important)

For different cameras, update the camera intrinsics used by visualization/tracking:

* Edit `cam_K.txt` in `Real_time_vis`

Make sure `cam_K.txt` matches your camera color intrinsics (fx, fy, cx, cy).

---

### C3) Local terminal — Run camera capture + frame sender (Azure Kinect DK)

On your host machine, open a **new terminal**, activate your Python environment, then run:

```bash
python Active/AzureKinectDK/real_time_get.py
```

After the window appears:

* Press **`s`** to select the object ROI in the first frame (initialization)
* Subsequent frames will keep tracking and streaming automatically

---

## D. Results & Permissions

BundleTrack inference results are saved to:

```
BundleTrack/results/AzureKinectDK/keyframes
```

Because this folder is written by docker, you may need to grant permissions on the host:

```bash
sudo chmod -R 777 ./*
```

---

## E. Reconstruction / Visualization

After tracking finishes, go to the **project root** (Inhand_Activate) and run:

```bash
python Active/viz.py
```

---

### Notes / Tips

* Real-time quality depends heavily on **accurate intrinsics** and **aligned depth-to-color**
* Both terminals (lf-net + BundleTrack docker) must stay running
* If you cannot access result files, check docker volume mapping and permissions first
