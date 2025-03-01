import cv2
import numpy as np
from tqdm import tqdm

from utils import Utils

import matplotlib.pyplot as plt


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def run_main(img_path, vid_path, colorize, krnl_size=3, white_background=True, edges_only=False, padding_frames=90):

    img = cv2.imread(img_path)
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # initialize image
    blank_img = np.zeros(img.shape).astype(np.float32)
    if white_background:
        blank_img = 255 * np.ones(img.shape).astype(np.float32)

    splitted_images = []
    for i in tqdm(range(0, 255, 1)):
        idxs = np.where(gray_scale < i)
        blank_img[idxs] = img[idxs]
        splitted_images.append(blank_img.astype(np.uint8))

    splitted_images = Utils.add_padding(splitted_images, img, padding_frames)
    splitted_images = Utils.add_smoothing_effect_to_img_sequence(splitted_images)
    Utils.write_to_video(img, splitted_images, vid_path, white_background, edges_only, colorize)



if __name__ == "__main__":

    img_path =""
    vid_path =""

    run_main(
        img_path,
        vid_path,
        colorize=(1.0, 1.0, 1.0),
        krnl_size=3,
        white_background=True,
        edges_only=False,
        padding_frames=90
    )
