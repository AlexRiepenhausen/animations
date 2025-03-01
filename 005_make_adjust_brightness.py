import os
import cv2
import glob
import numpy as np


def normalize(value_img):

    min_val, max_val = np.min(value_img), np.max(value_img)
    if max_val - min_val > 220.0:
        min_val, max_val = get_bounds(value_img, percentage=0.005)

    value_img = (value_img - min_val) / (max_val - min_val)
    value_img[value_img > 1.0] = 1.0
    value_img[value_img < 0.0] = 0.0

    return value_img


def get_bounds(value_img, percentage=0.01):
    hist, _ = np.histogram(value_img, bins=np.arange(0, 256))
    cdf = np.cumsum(hist) / np.sum(hist)
    lower_bound = np.min(np.where(cdf >= percentage)[0])
    upper_bound = np.min(np.where(cdf >= 1.0 - percentage)[0])
    return lower_bound, upper_bound


def apply_sine_func(value_img):
    scaling = 1.0 + 0.25 * np.sin(2.0 * np.pi * (value_img - 0.5))
    value_img = value_img * scaling
    value_img[value_img > 1.0] = 1.0
    value_img[value_img < 0.0] = 0.0
    return value_img


def adjust_brightness(
    input_folder_path, 
    output_folder_path
):

    file_names = glob.glob(input_folder_path + "*")
    for file_name in file_names:
        file_ext = file_name[-3:].lower()
        if file_ext in ["jpg", "png"]:

            new_file_name = output_folder_path + os.path.basename(file_name)
            print(new_file_name)

            img = cv2.imread(file_name).astype(np.uint8)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

            value_img = hsv[:,:,2]
            value_img = normalize(value_img)
            value_img = apply_sine_func(value_img)

            hsv[:,:,2] = 255 * value_img
            img_new = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            img = 0.5 * img.astype(np.float32) + 0.5 * img_new.astype(np.float32)

            cv2.imwrite(new_file_name, img.astype(np.uint8))


if __name__ == "__main__":

    # C:\Users\alexe\OneDrive\Desktop\Christmas\photo
    adjust_brightness(
        input_folder_path="",
        output_folder_path=""
    )
