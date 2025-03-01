import cv2
import numpy as np
import datetime

from collections import Counter


def get_area_rgb_vals(
    frame: np.ndarray,
    y0: int = 0,
    y1: int = 1080,
    x0: int = 0,
    x1: int = 1920
) -> list:
    frm = cv2.GaussianBlur(frame[y0:y1, x0:x1], (3, 3), 3)
    hist = np.reshape(frm, (frm.shape[0] * frm.shape[1], 3)).astype(np.uint8)
    tuples = [tuple(row) for row in hist]
    counter = Counter(tuples)
    most_common_tuple, _ = counter.most_common(1)[0]
    area_rgb_vals = [most_common_tuple[0], most_common_tuple[1], most_common_tuple[2]]
    return area_rgb_vals


def get_diff_mask(
    frame: np.ndarray,
    area_rgb_vals: list
) -> np.ndarray:
    diff_mask = np.max(np.abs(frame.astype(np.float32) - area_rgb_vals), axis=2)
    diff_mask = diff_mask / np.max(diff_mask)
    return diff_mask


def increase_saturation(frame: np.ndarray, hue_factor=0.95, saturation_factor=2.0):
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV) 
    image[:, :, 0] = image[:, :, 0] * hue_factor
    image[:, :, 1] = image[:, :, 1] * saturation_factor
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image.astype(np.float32)


def generate_mask(img: np.ndarray):

    mask = img[:,:,1] / (0.0001 + np.max([img[:,:,0], img[:,:,2]], axis=0))
    mask[mask >= 1.1] = 1.1
    mask = 231.81 * mask

    mask = cv2.medianBlur(mask.astype(np.uint8), 7).astype(np.float32)

    thresh = 233
    mask[mask <= thresh] = 1.0
    for i, pixel_val in enumerate(range(thresh + 8, thresh, -1)):
        opacity = (i / 9) ** 2.0
        mask[mask >= pixel_val] = opacity

    mask = cv2.GaussianBlur(mask, (3, 3), 3)
    mask = np.dstack([mask, mask, mask])

    return mask


def run_main(
    video_path: str, 
    background_path: str, 
    out_vid_path: str,
    hue_factor=0.99,
    saturation_factor=1.2
):

    cap_main = cv2.VideoCapture(video_path)
    cap_bkgr = cv2.VideoCapture(background_path)

    out = cv2.VideoWriter(
        out_vid_path, 
        cv2.VideoWriter_fourcc(*'mp4v'), 
        30, 
        (1920, 1080)
    )

    count = 0
    while(cap_main.isOpened()):

        ret_main, frame = cap_main.read()
        ret_bkgr, bkr_frame = cap_bkgr.read()

        if count % 30 == 0:
            t = datetime.datetime.now()
            print(f"second: {int(count / 30):5d}, frames: {count:5d}, time: {t}")

        if ret_main == False or ret_bkgr == False:
            break
        else:

            bkr_frame = bkr_frame.astype(np.float32)
            bkr_frame = cv2.resize(bkr_frame, (1920, 1080), cv2.INTER_CUBIC)

            tmp = increase_saturation(
                frame, 
                hue_factor=hue_factor,
                saturation_factor=saturation_factor
            )

            mask = generate_mask(tmp)
            out_frame = frame.astype(np.float32) * mask + bkr_frame * (1.0 - mask)

            out.write(out_frame.astype(np.uint8))

        count += 1

    cap_main.release()
    cap_bkgr.release()


if __name__ == "__main__":
    run_main(
        video_path="",
        background_path="",
        out_vid_path="",
        hue_factor=0.99, # 0.95
        saturation_factor=1.2 # 1.4
    )
