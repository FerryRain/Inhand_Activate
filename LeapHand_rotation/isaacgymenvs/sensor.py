import numpy as np
import serial
import threading
import cv2
import time
import torch
from scipy.ndimage import gaussian_filter
# import matplotlib.pyplot as plt
# import seaborn as sns
# os.system('cls')

contact_data_norm = np.zeros((16,16))
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 400
# WINDOW_WIDTH = contact_data_norm.shape[1]*30
# WINDOW_HEIGHT = contact_data_norm.shape[0]*30
cv2.namedWindow("Contact Data_left", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Contact Data_left",WINDOW_WIDTH, WINDOW_HEIGHT)
THRESHOLD =5
NOISE_SCALE =5

def readThread(serDev):
    global contact_data_norm, flag
    data_tac = []
    num = 0
    t1 = 0
    backup = None
    flag = False
    current = None


    EXPECTED_ROWS = 16
    EXPECTED_COLS = 16

    # --- Initialization phase: collect many frames to compute median ---
    while True:
        if serDev.in_waiting > 0:
            try:
                line = serDev.readline().decode('utf-8').strip()
            except Exception:
                line = ""
            if len(line) < 10:
                if current is not None and len(current) == EXPECTED_ROWS:
                    try:
                        arr = np.asarray(current, dtype=float)
                    except Exception as e:
                        print("Warning: cannot convert current to float array:", e)
                        current = []
                        continue
                    if arr.shape == (EXPECTED_ROWS, EXPECTED_COLS):
                        backup = arr.copy()
                        if t1 != 0:
                            print("fps", 1.0 / (time.time() - t1))
                        t1 = time.time()
                        data_tac.append(backup)
                        num += 1
                        if num > 30:
                            break
                    else:
                        print("Skipped frame with unexpected shape:", arr.shape)
                current = []
                continue

            if current is not None:
                str_values = line.split()
                try:
                    int_values = [int(val) for val in str_values]
                except ValueError:
                    current.append([0] * EXPECTED_COLS)
                    continue
                if len(int_values) != EXPECTED_COLS:
                    if len(int_values) < EXPECTED_COLS:
                        int_values = int_values + [0] * (EXPECTED_COLS - len(int_values))
                    else:
                        int_values = int_values[:EXPECTED_COLS]
                current.append(int_values)
    if len(data_tac) == 0:
        raise RuntimeError("No valid frames collected for median computation.")
    try:
        data_tac_stack = np.stack(data_tac, axis=0)   # shape (N,16,16)
    except Exception as e:
        print("Error stacking frames for median:", e)
        for i, f in enumerate(data_tac):
            print(i, type(f), getattr(f, "shape", None))
        raise

    median = np.median(data_tac_stack, axis=0).astype(float)  # shape (16,16)
    flag = True
    print("Finish Initialization!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    while True:
        if serDev.in_waiting > 0:
            try:
                line = serDev.readline().decode('utf-8').strip()
            except Exception:
                line = ""
            if len(line) < 10:
                if current is not None and len(current) == EXPECTED_ROWS:
                    try:
                        backup = np.asarray(current, dtype=float)
                    except Exception as e:
                        print("Warning: cannot convert current to float array (runtime):", e)
                        backup = None
                current = []
                if backup is not None:
                    if not isinstance(backup, np.ndarray):
                        backup = np.asarray(backup, dtype=float)
                    if not isinstance(median, np.ndarray):
                        median = np.asarray(median, dtype=float)

                    if backup.shape == median.shape:
                        try:
                            contact_data = backup - median - THRESHOLD
                        except Exception as e:
                            print("Error subtracting arrays:", e)
                            contact_data = np.asarray([[float(backup[i,j]) - float(median[i,j]) - THRESHOLD
                                                        for j in range(min(backup.shape[1], median.shape[1]))]
                                                       for i in range(min(backup.shape[0], median.shape[0]))], dtype=float)
                    else:
                        min_rows = min(backup.shape[0], median.shape[0])
                        min_cols = min(backup.shape[1], median.shape[1])
                        contact_data = backup[:min_rows, :min_cols] - median[:min_rows, :min_cols] - THRESHOLD
                        if contact_data.shape != (EXPECTED_ROWS, EXPECTED_COLS):
                            padded = np.zeros((EXPECTED_ROWS, EXPECTED_COLS), dtype=float)
                            padded[:contact_data.shape[0], :contact_data.shape[1]] = contact_data
                            contact_data = padded

                    contact_data = np.clip(contact_data, 0, 100)

                    if np.max(contact_data) < THRESHOLD:
                        contact_data_norm = contact_data / NOISE_SCALE
                    else:
                        contact_data_norm = contact_data / np.max(contact_data)

                continue

            if current is not None:
                str_values = line.split()
                try:
                    int_values = [int(val) for val in str_values]
                except ValueError:
                    int_values = [0] * EXPECTED_COLS
                if len(int_values) != EXPECTED_COLS:
                    if len(int_values) < EXPECTED_COLS:
                        int_values = int_values + [0] * (EXPECTED_COLS - len(int_values))
                    else:
                        int_values = int_values[:EXPECTED_COLS]
                current.append(int_values)
                continue



# PORT = "left_gripper_right_finger"
PORT ='/dev/ttyUSB0'
BAUD = 2000000
# serDev = serial.Serial(PORT,2000000)
serDev = serial.Serial('/dev/ttyUSB0',BAUD)
exitThread = False
serDev.flush()
serialThread = threading.Thread(target=readThread, args=(serDev,))
serialThread.daemon = True
serialThread.start()


def apply_gaussian_blur(contact_map, sigma=0.1):
    return gaussian_filter(contact_map, sigma=sigma)

def temporal_filter(new_frame, prev_frame, alpha=0.2):
    """
    Apply temporal smoothing filter.
    'alpha' determines the blending factor.
    A higher alpha gives more weight to the current frame, while a lower alpha gives more weight to the previous frame.
    """
    return alpha * new_frame + (1 - alpha) * prev_frame

# Initialize previous frame buffer
prev_frame = np.zeros_like(contact_data_norm)

if __name__ == '__main__':

    print('receive data test')

    while True:

        for i in range(300):
            if flag:
                temp_filtered_data = temporal_filter(contact_data_norm, prev_frame)
                prev_frame = temp_filtered_data

                # Scale to 0-255 and convert to uint8
                temp_filtered_data_scaled = (temp_filtered_data * 255).astype(np.uint8)
                data = temp_filtered_data_scaled[8:12, 12:16]
                data = data.reshape(16,order='F')
                data = data[[8, 6, 1, 10, 7, 4, 2, 9, 5, 3, 0, 11, 12, 15, 13, 14]]
                data_n = np.where(data>50,1,0)
                print(data_n)
                # Apply color map
                colormap = cv2.applyColorMap(data, cv2.COLORMAP_VIRIDIS)

                cv2.imshow("Contact Data_left", colormap)
                cv2.waitKey(1)
            time.sleep(0.01)

