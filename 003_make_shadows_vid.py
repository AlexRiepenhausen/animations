import cv2
import datetime
import numpy as np


def add_shadows_to_frame(img, offset_y, offset_x, brightness, width, blur, alpha):

    dim_y, dim_x = img.shape[0], img.shape[1]

    # add shadow
    def adjust_for_overflow(shadow_items, max_dim):
        underflow_idxs = np.where(shadow_items < 0)
        overflow_idxs = np.where(shadow_items >= max_dim)
        shadow_items[underflow_idxs] = 0
        shadow_items[overflow_idxs] = max_dim - 1
        return shadow_items

    # get starting points of shadow
    item_idxs = np.where(img < 230)
    item_idxs_y = adjust_for_overflow(item_idxs[0] + offset_y, dim_y)
    item_idxs_x = adjust_for_overflow(item_idxs[1] + offset_x, dim_x)
    item_idxs_z = item_idxs[2]

    # init shadow image
    shadow_idxs = (item_idxs_y, item_idxs_x, item_idxs_z)
    shadow_img = np.full(img.shape, 255)
    shadow_img[shadow_idxs] = brightness

    # apply blur
    shadow_img = cv2.GaussianBlur(shadow_img.astype(np.uint8), (blur, blur), 3)
    shadow_img = cv2.GaussianBlur(shadow_img.astype(np.uint8), (blur, blur), 3)

    return shadow_img


def add_shadows(
    in_vid_path,
    out_vid_path,
    offset_y=6,
    offset_x=6,
    brightness=150,
    width=5,
    blur=5,
    alpha=0.2
):

    cap = cv2.VideoCapture(in_vid_path)
    out = None

    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    count = 0
    while(cap.isOpened()):

        ret, frame = cap.read()

        if count % 30 == 0:
            print(f"second: {int(count / 30):5d}, frames: {count:5d}, time: {datetime.datetime.now()}")

        if ret == False:
            break
        else:

            if out is None:
                out = cv2.VideoWriter(
                    out_vid_path, 
                    cv2.VideoWriter_fourcc(*'mp4v'), 
                    30, 
                    (frame.shape[1], frame.shape[0])
                )

            frame = frame.astype(np.float32)
            shadow_img = add_shadows_to_frame(
                frame, 
                offset_y, 
                offset_x, 
                brightness, 
                width, 
                blur, 
                alpha
            )

            out.write(shadow_img.astype(np.uint8))

        count += 1

    cap.release()


if __name__ == "__main__":

    i = 0

    in_vid_path = ""
    out_vid_path = ""

    add_shadows(
        in_vid_path,
        out_vid_path,
        offset_y=6,
        offset_x=6,
        brightness=150,
        width=5,
        blur=5,
        alpha=0.2
    )
