import cv2
import numpy as np
from tqdm import tqdm
from utils import Utils


class DynamicDrawing:

    def __init__(
        self, 
        kernel_size=1, 
        frames_per_area=5, 
        min_step_size=5, 
        padding_frames=90
    ):
        self.kernel_size = kernel_size
        self.frames_per_area = frames_per_area
        self.min_step_size = min_step_size
        self.padding_frames = padding_frames


    def draw(self, img_path, vid_path):

        img = cv2.imread(img_path).astype(np.float32)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.filter2D(gray_img, -1, Utils.define_flat_kernel(self.kernel_size))

        background_color = self._init_background_color(gray_img)
        binary_img = self._init_binary_image(gray_img, background_color)
        unique_labels, labels_im = self._run_segmentation(binary_img, background_color)

        # initialize image
        blank_img = np.zeros(img.shape).astype(np.float32)
        if background_color == 255:
            blank_img = 255 * np.ones(img.shape).astype(np.float32)

        splitted_images = []
        for lbl in unique_labels:

            mask = self._init_mask(labels_im, lbl)
            y_indices = self._init_y_movement_info(mask)

            for idx0, idx1 in y_indices:
                ys, xs = np.where(mask[idx0:idx1] > 0.5)
                idxs = (ys + idx0, xs)
                blank_img[idxs] = img[idxs]
                splitted_images.append(blank_img.astype(np.uint8))

            if len(y_indices) == 0:
                idxs = np.where(mask > 0.5)
                blank_img[idxs] = img[idxs]
                splitted_images.append(blank_img.astype(np.uint8))

        splitted_images = Utils.add_padding(splitted_images, img, self.padding_frames)
        splitted_images = Utils.add_smoothing_effect_to_img_sequence(splitted_images)
        self._write_to_video(img, splitted_images, vid_path)


    def _init_y_movement_info(self, mask):

        vertical = np.sum(mask, axis=1)
        ys = np.where(vertical > 0.0)[0]

        ys_start, y_end = np.min(ys), np.max(ys)
        distance = y_end - ys_start

        step_size = 0
        if distance >= self.frames_per_area:
            step_size = distance // self.frames_per_area
            if step_size < self.min_step_size:
                step_size = self.min_step_size
        else:
            return []

        split_points = [i for i in range(ys_start, y_end, step_size)]
        if split_points[-1] < y_end:
            split_points += [y_end + 1]

        y_indices = []
        for i in range(1, len(split_points)):
            idx0, idx1 = split_points[i - 1], split_points[i]
            y_indices.append((idx0, idx1))

        return y_indices


    def _run_segmentation(self, binary_img, background_color):

        # find connected components
        retval, labels_im, stats, centroids = cv2.connectedComponentsWithStats(
            binary_img.astype(np.uint8)
        )

        # sort by number of items
        unique_labels, counts = np.unique(labels_im, return_counts=True)
        idxs = np.argsort(counts)[::-1]
        unique_labels = unique_labels[idxs]

        # sort top to bottom
        y_positions = []
        for lbl in unique_labels:
            top_most_y = np.min(np.where(labels_im == lbl)[0])
            y_positions.append(top_most_y)

        idxs = np.argsort(y_positions)
        unique_labels = unique_labels[idxs]

        return unique_labels, labels_im


    def _init_mask(self, labels_im, lbl):
        mask = np.zeros(labels_im.shape)
        idxs = np.where(labels_im == lbl)
        mask[idxs] = 1.0
        return mask


    def _init_binary_image(self, gray_img, background_color):
        if background_color == 0:
            binary_img = cv2.threshold(gray_img, 20, 255, cv2.THRESH_BINARY)[1]
            return binary_img
        else:
            binary_img = cv2.threshold(gray_img, 235, 255, cv2.THRESH_BINARY_INV)[1]
            return binary_img


    def _init_background_color(self, gray_img):
        counts, items = np.histogram(gray_img, bins=255)
        num_blacks, num_whites = np.sum(counts[:20]), np.sum(counts[235:])
        background_color = 0 if num_blacks >= num_whites else 255
        return background_color


    def _write_to_video(self, img, splitted_images, vid_path):
        num_frames = len(splitted_images)
        out = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (img.shape[1], img.shape[0]))
        for i in tqdm(range(num_frames)):
            out.write(splitted_images[i])  
        out.release()


if __name__ == "__main__":

    img_path = ""
    bkr_path = ""
    vid_path = ""

    dd = DynamicDrawing(
        kernel_size=3, 
        frames_per_area=3, 
        min_step_size=1, 
        padding_frames=30
    )

    dd.draw(img_path, vid_path)

    # simple img params (only a few groups): 
    # frames_per_area=30, 
    # min_step_size=5, 
    # padding_frames=90

    # realistic drawing params (many groups): 
    # frames_per_area=30, 
    # min_step_size=20,
    # padding_frames=90
