import cv2
import mss
import numpy as np


def run_main(file_path: str):
    with mss.mss() as sct:
        mon = {'left': 0, 'top': 0, 'width': 1920, 'height': 1080}
        img = np.array(sct.grab(mon))
        cv2.imwrite(file_path, img.astype(np.uint8))


if __name__ == "__main__":
    run_main(
        file_path=""
    )
