import cv2
import numpy as np


def add_shadows(img, offset_y=10, offset_x=10, brightness=0, width=5, blur=9, alpha=0.5):

    dim_y, dim_x = img.shape[0], img.shape[1]

    # add shadow
    def adjust_for_overflow(shadow_items, max_dim):
        underflow_idxs = np.where(shadow_items < 0)
        overflow_idxs = np.where(shadow_items >= max_dim)
        shadow_items[underflow_idxs] = 0
        shadow_items[overflow_idxs] = max_dim - 1
        return shadow_items

    # get starting points of shadow
    item_idxs = np.where(img < 255)
    item_idxs_y = adjust_for_overflow(item_idxs[0] + offset_y, dim_y)
    item_idxs_x = adjust_for_overflow(item_idxs[1] + offset_x, dim_x)
    item_idxs_z = item_idxs[2]

    # init shadow image
    shadow_idxs = (item_idxs_y, item_idxs_x, item_idxs_z)
    shadow_img = np.full(img.shape, 255)
    shadow_img[shadow_idxs] = brightness

    # apply width
    shadow_img = cv2.GaussianBlur(shadow_img.astype(np.uint8), (width, width), 3)
    idxs = np.where(shadow_img < 255)
    shadow_img[idxs] = brightness

    # apply blur
    shadow_img = cv2.GaussianBlur(shadow_img.astype(np.uint8), (blur, blur), 3)
    shadow_img = cv2.GaussianBlur(shadow_img.astype(np.uint8), (blur, blur), 3)

    # superimpose image on shadow
    background_idxs = np.where(img == 255)
    item_idxs = np.where(img < 255)

    shadow_img[background_idxs] = 255 * (1.0 - alpha) + shadow_img[background_idxs] * alpha
    shadow_img[item_idxs] = img[item_idxs]

    return shadow_img


if __name__ == "__main__":

    img_path = ""
    out_path = ""

    img = cv2.imread(img_path).astype(np.float32)

    shadow_img = add_shadows(
        img,
        offset_y=6,
        offset_x=6,
        brightness=150,
        width=5,
        blur=5,
        alpha=0.2
    )

    cv2.imwrite(out_path, shadow_img.astype(np.uint8))
