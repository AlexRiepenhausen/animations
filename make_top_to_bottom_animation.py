import cv2
import numpy as np
from tqdm import tqdm
from typing import Union, List

from utils import Utils


class TopToBottom:

    def __init__(
        self, 
        kernel_size, 
        padding_front_color,
        padding_frames_front, 
        padding_frames_back, 
        frame_repetition=1
    ):
        self.kernel_size = kernel_size
        self.padding_front_color = padding_front_color
        self.padding_frames_front = padding_frames_front
        self.padding_frames_back = padding_frames_back
        self.frame_repetition = frame_repetition


    def draw_to_vid_file(
        self, 
        img_path: str = "", 
        bkr_path: str = "", 
        vid_path: str = ""
    ):
        img = cv2.imread(img_path).astype(np.float32)
        bkr_img = None
        if img_path != bkr_path:
            bkr_img = cv2.imread(bkr_path).astype(np.float32)
        splitted_images = self.draw_to_array(img, bkr_img)
        self._write_to_video(img, splitted_images, vid_path)


    def draw_to_array(
        self, 
        img: np.ndarray,
        bkr_img: Union[np.ndarray, None] = None
    ) -> List:

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.filter2D(gray_img, -1, Utils.define_flat_kernel(self.kernel_size))

        # init background color and binary image
        background_color = self._init_background_color(gray_img)
        binary_img = self._init_binary_image(gray_img, background_color)

        if bkr_img is None:
            bkr_img = np.full(img.shape, background_color)

        # segmentation process
        _, labels_im, _, centroids = cv2.connectedComponentsWithStats(binary_img.astype(np.uint8))
        tiered_labels = self._sorting_process(labels_im, centroids)

        # generate frames
        splitted_images, canvas_img = self._split_images_sorted(img, bkr_img, labels_im, tiered_labels, background_color)
        splitted_images = Utils.add_padding_front(splitted_images, img, self.padding_frames_front, self.padding_front_color)
        splitted_images = Utils.add_padding(splitted_images, canvas_img, self.padding_frames_back)
        splitted_images = Utils.add_smoothing_effect_to_img_sequence(splitted_images)
        splitted_images = Utils.add_smoothing_effect_to_img_sequence(splitted_images)
        splitted_images = Utils.add_smoothing_effect_to_img_sequence(splitted_images)

        return splitted_images


    def _init_background_color(self, gray_img):
        counts, items = np.histogram(gray_img, bins=255)
        num_blacks, num_whites = np.sum(counts[:20]), np.sum(counts[235:])
        background_color = 0 if num_blacks >= num_whites else 255
        return background_color


    def _init_binary_image(self, gray_img, background_color):
        if background_color == 0:
            binary_img = cv2.threshold(gray_img, 20, 255, cv2.THRESH_BINARY)[1]
            return binary_img
        else:
            binary_img = cv2.threshold(gray_img, 235, 255, cv2.THRESH_BINARY_INV)[1]
            return binary_img


    def _write_to_video(self, img, splitted_images, vid_path):
        num_frames = len(splitted_images)
        out = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (img.shape[1], img.shape[0]))
        for i in tqdm(range(num_frames)):
            out.write(splitted_images[i])  
        out.release()


    def _sorting_process(self, labels_im, centroids):

        unique_labels, counts = np.unique(labels_im, return_counts=True)

        # sort by frequency first
        idxs = np.argsort(counts)[::-1]
        unique_labels = unique_labels[idxs]
        counts = counts[idxs]
        centroids = centroids[idxs]

        # split background(s) from actual items
        split_point = np.mean(counts)
        max_background_id = np.max(np.where(counts >= split_point)[0]) + 1
        start_labels = [int(unique_labels[i]) for i in range(max_background_id)]

        # remove items after splitting background
        unique_labels = unique_labels[max_background_id:]
        counts = counts[max_background_id:]
        centroids = centroids[max_background_id:]

        # left to right top to bottom
        ys = centroids[:, 1]

        # generate y tiers
        idxs = np.argsort(ys)[::-1]
        temp_ys = ys[idxs]

        differentials = np.sqrt(np.abs(np.diff(temp_ys, prepend=0.0)))
        mean_values = np.median(differentials)
        top_n_maxes = np.mean(np.sort(differentials)[::-1][:10])
        split_point = mean_values + 0.5 * (top_n_maxes - mean_values)

        idxs = np.where(differentials >= split_point)[0]
        y_tiers = temp_ys[idxs]

        if np.max(temp_ys) not in y_tiers:
            y_tiers = np.concatenate(([np.max(temp_ys)], y_tiers))

        if np.min(temp_ys) not in y_tiers:
            y_tiers = np.concatenate((y_tiers, [np.min(temp_ys)]))

        tiered_idxs = []
        tiered_x_vals = []
        num_item = len(y_tiers)

        for i in range(1, num_item):
            high, low = y_tiers[i - 1], y_tiers[i]
            idxs = np.where((ys <= high) & (ys >= low))[0]
            tiered_idxs.append(idxs)
            tiered_x_vals.append(centroids[idxs][:,0])

        tiered_labels = []
        for i in range(len(tiered_idxs)):

            # sort selected global idxs by the x value
            xs = tiered_x_vals[i]
            sorted_idxs = np.argsort(xs)[::-1]
            global_idxs = tiered_idxs[i][sorted_idxs][::-1]

            # map global indices to unique labels
            tiered_labels.append(unique_labels[global_idxs])

        tiered_labels = tiered_labels[::-1]
        tiered_labels = [start_labels] + tiered_labels

        return tiered_labels

    def _split_images_sorted(self, img, bkr_img, labels_im, tiered_labels, background_color):

        splitted_images = []
        mask = np.zeros(labels_im.shape)
        splitted_images.append(bkr_img.astype(np.uint8))

        canvas_img = np.copy(bkr_img)

        for i, tier_labels in tqdm(enumerate(tiered_labels[1:])):

            for lbl in tier_labels:

                mask[labels_im == lbl] = 1.0
                idxs = np.where(mask == 1.0)
                canvas_img[idxs] = img[idxs]

                splitted_images.append(canvas_img.astype(np.uint8))

                if i == 0:
                    for _ in range(3):
                        splitted_images.append(canvas_img.astype(np.uint8))
                else:
                    for _ in range(self.frame_repetition):
                        splitted_images.append(canvas_img.astype(np.uint8))

        return splitted_images, canvas_img


if __name__ == "__main__":

    img_path = ""
    bkr_path = ""
    vid_path = ""

    ttb = TopToBottom(
        kernel_size=3,
        padding_front_color=(255, 255, 255),
        padding_frames_front=1,
        padding_frames_back=120,
        frame_repetition=3
    )

    ttb.draw_to_vid_file(
        img_path, 
        bkr_path, 
        vid_path
    )
