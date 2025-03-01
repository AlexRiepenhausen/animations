import cv2
import numpy as np

from enum import Enum
from tqdm import tqdm

from utils import Utils


class Direction(Enum):
    L2R = 0
    R2L = 1
    U2D = 2
    D2U = 3


def write_to_video(img, splitted_images, vid_path):
    num_frames = len(splitted_images)
    out = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (img.shape[1], img.shape[0]))
    for i in tqdm(range(num_frames)):
        out.write(splitted_images[i])  
    out.release()


def run_main(
    img_path,
    background_path,
    vid_path,
    direction=Direction.U2D,
    num_frames=60
):

    processed_frames = []

    img = cv2.imread(img_path).astype(np.float32)
    background_img = np.full(img.shape, 255)
    if background_path is not None:
        background_img = cv2.imread(background_path).astype(np.float32)

    y_dim, x_dim = img.shape[0], img.shape[1]
    processed_frames.append(background_img.astype(np.uint8))
    blank_img = np.copy(background_img)

    if direction == Direction.U2D:

        for i in range(num_frames):

            w = (i + 1) / num_frames
            y_idx0 = int(w * y_dim)

            mask = np.random.uniform(0.0, 0.2, img[:y_idx0].shape)
            blank_img[:y_idx0] = img[:y_idx0] * mask + blank_img[:y_idx0] * (1.0 - mask)
            processed_frames.append(blank_img.astype(np.uint8))

    # add one second
    for _ in range(30):
        mask = np.random.uniform(0.0, 0.2, img[:y_idx0].shape)
        blank_img = img * mask + blank_img * (1.0 - mask)
        processed_frames.append(blank_img.astype(np.uint8))
    processed_frames.append(img.astype(np.uint8))
    processed_frames.append(img.astype(np.uint8))

    processed_frames = Utils.add_smoothing_effect_to_img_sequence(processed_frames)
    write_to_video(img, processed_frames, vid_path)


if __name__ == "__main__":

    img_path = ""
    background_path = None
    vid_path = ""

    run_main(
        img_path,
        background_path,
        vid_path,
        direction=Direction.U2D,
        num_frames=30
    )
