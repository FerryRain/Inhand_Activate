BUNDLETRACK_DIR="/home/ferry/data/Code2/Research/Inhand_Activate/BundleTrack"
NOCS_DIR="/home/ferry/data/Code2/Research/Inhand_Activate/BundleTrack/NOCS"
YCBINEOAT_DIR="/home/ferry/data/Code2/Research/PoseEstimation/BundleTrack/iros_submission_version"
echo "BUNDLETRACK_DIR $BUNDLETRACK_DIR"
echo "NOCS_DIR $NOCS_DIR"
echo "YCBINEOAT_DIR $YCBINEOAT_DIR"

#docker run --gpus all -it --network=host --name bundletrack  -m  16000m --cap-add=SYS_PTRACE --security-opt seccomp=unconfined  -v $BUNDLETRACK_DIR:$BUNDLETRACK_DIR:rw -v $NOCS_DIR:$NOCS_DIR -v $YCBINEOAT_DIR:$YCBINEOAT_DIR -v /tmp:/tmp  --ipc=host -e DISPLAY=${DISPLAY} -e GIT_INDEX_FILE wenbowen123/bundletrack:3090 bash
docker run --gpus all -it --network=host --name bundletrack  -m  16000m --cap-add=SYS_PTRACE --security-opt seccomp=unconfined  -v $BUNDLETRACK_DIR:$BUNDLETRACK_DIR:rw -v $NOCS_DIR:$NOCS_DIR -v $YCBINEOAT_DIR:$YCBINEOAT_DIR -v /tmp:/tmp  --ipc=host -e DISPLAY=${DISPLAY} -e QT_X11_NO_MITSHM=1 -v /home/ferry/data/tmp/.X11-unix:/home/ferry/data/tmp/.X11-unix -e GIT_INDEX_FILE wenbowen123/bundletrack:3090 bash
