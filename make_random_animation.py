import cv2
import numpy as np
from tqdm import tqdm

from utils import Utils


class RandomAnimation:

    def __init__(self, kernel_size, padding_frames):
        self.kernel_size = kernel_size
        self.padding_frames = padding_frames


    def draw(self, img_path, vid_path):

        img = cv2.imread(img_path).astype(np.float32)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.filter2D(gray_img, -1, Utils.define_flat_kernel(self.kernel_size))

        # init background color and binary image
        background_color = self._init_background_color(gray_img)
        binary_img = self._init_binary_image(gray_img, background_color)
        unique_labels, labels_im = self._run_segmentation(binary_img)

        # generate frames
        splitted_images = self._split_images_random(img, labels_im, unique_labels, background_color)
        splitted_images = Utils.add_padding(splitted_images, img, self.padding_frames)
        splitted_images = Utils.add_smoothing_effect_to_img_sequence(splitted_images)
        self._write_to_video(img, splitted_images, vid_path)



    def _split_images_random(self, img, labels_im, unique_labels, background_color):

        splitted_images = []

        # initialize image
        blank_img = np.zeros(img.shape).astype(np.float32)
        if background_color == 255:
            blank_img = 255 * np.ones(img.shape).astype(np.float32)

        mask = np.zeros(labels_im.shape)
        splitted_images.append(blank_img.astype(np.uint8))

        for i, lbl in tqdm(enumerate(unique_labels)):

            mask[labels_im == lbl] = 1.0

            blank_img[:, :, 0] = mask * img[:, :, 0] + (1.0 - mask) * blank_img[:, :, 0]
            blank_img[:, :, 1] = mask * img[:, :, 1] + (1.0 - mask) * blank_img[:, :, 1]
            blank_img[:, :, 2] = mask * img[:, :, 2] + (1.0 - mask) * blank_img[:, :, 2]

            splitted_images.append(blank_img.astype(np.uint8))
        splitted_images.append(img.astype(np.uint8))  

        return splitted_images


    def _run_segmentation(self, binary_img):
        _, labels_im, _, _ = cv2.connectedComponentsWithStats(binary_img.astype(np.uint8))
        unique_labels = np.unique(labels_im)
        np.random.shuffle(unique_labels)
        return unique_labels, labels_im


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


if __name__ == "__main__":

    for i in range(1, 7):

        img_path = ""
        vid_path = ""

        rnda = RandomAnimation(kernel_size=1, padding_frames=140)
        rnda.draw(img_path, vid_path)
