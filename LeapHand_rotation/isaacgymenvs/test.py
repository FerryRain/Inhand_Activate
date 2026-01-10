import os
import queue
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from isaacgymenvs.hand_controller import HardwarePlayer
import cv2
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_name='config', config_path='cfg')
def main(config: DictConfig):
    agent = None
    WINDOW_NAME = "Contact Data_left"
    try:
        agent = HardwarePlayer(config)
        agent.restore_all_models()
        agent.start_deployment()

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 400, 400)

        agent.get_axis('z')
        while True:
            try:
                image = agent.image_queue.get_nowait()
                cv2.imshow(WINDOW_NAME, image)
            except queue.Empty:
                pass     
            key = cv2.waitKey(20) & 0xFF  # Use a small delay like 20ms
            if key == ord('q'):
                print("[Main Thread] 'q' key pressed. Shutting down.")
                break
            if not agent._deploy_thread.is_alive():
                print("[Main Thread] Deployment thread has unexpectedly stopped. Exiting.")
                break
            agent.get_axis()


    except KeyboardInterrupt:
        print("\n[Main Thread] Ctrl+C detected. Shutting down all threads gracefully...")

    finally:
        cv2.destroyAllWindows()
        if agent:
            agent.stop()
        print("[Main Thread] Program has exited cleanly.")


if __name__ == '__main__':
    main()