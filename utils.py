import cv2
import numpy as np
from tqdm import tqdm
import scipy.stats as st

class Utils:

    # smoothing and sharpening
    KRNL01 = np.array(
        [
            [-0.1, 0.2, -0.1], 
            [ 0.2, 0.6,  0.2], 
            [-0.1, 0.2, -0.1]
        ]
    )

    KRNL02 = np.array(
        [
            [0.2/8.0, 0.2/8.0, 0.2/8.0], 
            [0.2/8.0, 0.8    , 0.2/8.0], 
            [0.2/8.0, 0.2/8.0, 0.2/8.0]
        ]
    )

    KRNL03 = np.array(
        [
            [0.125, 0.125, 0.125], 
            [0.125, -1.0 , 0.125], 
            [0.125, 0.125, 0.125]
        ]
    )

    @staticmethod
    def standardize(img):
        mean_val, stdev = np.mean(img), np.std(img)
        return (img - mean_val) / stdev

    @staticmethod
    def normalize(img):
        minimum, maximum = np.min(img), np.max(img)
        return (img - minimum) / (maximum - minimum)

    @staticmethod
    def smoothing_kernel_gauss(dim=3, nsig=3):
        x = np.linspace(-nsig, nsig, dim+1)
        kern1d = np.diff(st.norm.cdf(x))
        kern2d = np.outer(kern1d, kern1d)
        return kern2d / kern2d.sum()

    @staticmethod
    def define_flat_kernel(krnl_size):
        return np.ones((krnl_size, krnl_size)) / (krnl_size ** 2.0)

    @staticmethod
    def zoom_at(img, zoom=1.01, coord=None):

        h, w, _ = [ zoom * i for i in img.shape ]

        if coord is None: cx, cy = w/2, h/2
        else: cx, cy = [ zoom*c for c in coord ]
        
        img = cv2.resize( img, (0, 0), fx=zoom, fy=zoom)
        img = img[ int(round(cy - h/zoom * .5)) : int(round(cy + h/zoom * .5)),
                int(round(cx - w/zoom * .5)) : int(round(cx + w/zoom * .5)),
                : ]
        
        return img
    
    @staticmethod
    def add_padding_front(splitted_images, img, padding_frames, padding_color):
        padding_img = np.full((img.shape), padding_color)
        padding_front = []
        for _ in range(padding_frames):
            padding_front.append(padding_img.astype(np.uint8))
        splitted_images = padding_front + splitted_images
        return splitted_images

    @staticmethod
    def add_padding(splitted_images, img, padding_frames):
        for _ in range(padding_frames):
            splitted_images.append(img)
        return splitted_images

    @staticmethod
    def write_to_video(img, splitted_images, vid_path, white_background, edges_only, colorize):

        def colorize_image(img, colorize):
            img = img.astype(np.float32)
            img[:, :, 0] *= colorize[0]
            img[:, :, 1] *= colorize[1]
            img[:, :, 2] *= colorize[2]
            img = img.astype(np.uint8)
            return img

        if white_background:
            if edges_only:
                for i in range(len(splitted_images)):
                    splitted_images[i] = np.abs(splitted_images[i] - 255)
                    splitted_images[i] = np.abs(255 - splitted_images[i].astype(np.float32)).astype(np.uint8)
        else:
            for i in range(len(splitted_images)):
                if edges_only:
                    splitted_images[i] = np.abs(splitted_images[i] - 255)
                    splitted_images[i] = colorize_image(splitted_images[i], colorize)
                else:
                    splitted_images[i] = np.abs(splitted_images[i].astype(np.float32) - 255.0).astype(np.uint8)
                    splitted_images[i] = colorize_image(splitted_images[i], colorize)

        num_frames = len(splitted_images)
        out = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (img.shape[1], img.shape[0]))
        for i in tqdm(range(num_frames)):
            out.write(splitted_images[i])  
        out.release()

    @staticmethod
    def add_smoothing_effect_to_img_sequence(splitted_images):
        for i in tqdm(range(1, len(splitted_images))):
            merged = 0.5 * splitted_images[i - 1] + 0.5 * splitted_images[i]
            splitted_images[i] = merged.astype(np.uint8)
        return splitted_images

    @staticmethod
    def sigmoid_scaling_mask(mask):
        idxs0 = np.where(mask < 0.5)
        idxs1 = np.where(mask >= 0.5)
        mask[idxs0] = 2.0 * mask[idxs0] ** 2.0
        mask[idxs1] = 1.0 - 2.0 * (mask[idxs1] - 1.0) ** 2.0
        return mask
