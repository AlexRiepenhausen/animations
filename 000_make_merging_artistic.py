import cv2
import numpy as np
from tqdm import tqdm
import random
from utils import Utils

# applies sharpening with kernels defined in utils
def sharpening(img):
    img_krnl0 = cv2.filter2D(img, -1, Utils.KRNL01)
    img_krnl2 = cv2.filter2D(img, -1, Utils.KRNL03)
    img = img_krnl0 - Utils.standardize(img_krnl2)
    img[img > 255] = 255
    img[img < 0] = 0
    return img


def get_file_names(path_to_frames):
    file_names = []
    for i in range(0, 19):
        file_name = path_to_frames + f"m{i:03d}.png"
        file_names.append(file_name)
    return file_names


def get_frames(file_names):
    frames = []
    for img_path in file_names:
        img = cv2.imread(img_path).astype(np.uint8)
        frames.append(img)
    return frames


def write_to_video(processed_frames, vid_path):
    dims = (processed_frames[0].shape[1], processed_frames[0].shape[0])
    num_frames = len(processed_frames)
    out = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, dims)
    for i in tqdm(range(num_frames)):
        out.write(processed_frames[i])  
    out.release()


def run_main(path_to_frames, vid_path, transition_frames=24, duplication_factor=8, trippy=True):

    file_names = get_file_names(path_to_frames)
    input_frames = get_frames(file_names)

    def sigmoid(x):
        y = 1.0 / (1.0 + np.exp(-5.0*x + 2.5))
        return y

    def get_sim_mask(current_frame, previous_frame):

        blur_krnl = np.ones((49, 49)) / (49 ** 2.0)

        current_frame_gry = cv2.filter2D(cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY), -1, blur_krnl)
        previous_frame_gry = cv2.filter2D(cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY), -1, blur_krnl)

        sim_mask_gr = -(current_frame_gry - previous_frame_gry)

        sim_mask = np.ones(current_frame.shape)
        sim_mask[:,:,0] = sim_mask_gr
        sim_mask[:,:,1] = sim_mask_gr
        sim_mask[:,:,2] = sim_mask_gr

        if np.max(sim_mask) == 0.0:
            return np.ones(current_frame.shape)

        sim_mask = 1.0 - Utils.normalize(sim_mask)

        return sim_mask

    processed_frames = [input_frames[0] for _ in range(transition_frames)]

    for i in tqdm(range(1, len(input_frames))):

        current_frame = input_frames[i].astype(np.float32)
        current_frame_sharp = sharpening(sharpening(current_frame))

        # initial comparison frame
        initial_idx = -transition_frames
        previous_frame = processed_frames[initial_idx].astype(np.float32)
        sim_mask = get_sim_mask(current_frame, previous_frame)

        for j in range(transition_frames):

            idx = -(transition_frames - j)
            weight = (j / transition_frames) ** 2.0

            sim_mask = Utils.zoom_at(sim_mask, zoom=1.0 + 0.01)

            if trippy:

                previous_frame = previous_frame * (1.0 - sim_mask * weight) + current_frame * sim_mask * weight
                current_frame = Utils.zoom_at(current_frame, zoom=1.0 + 0.01 * weight)

                out_frame = current_frame * (1.0 - weight) + previous_frame * weight
                processed_frames[idx] = out_frame.astype(np.uint8)

            else:

                previous_frame = previous_frame * (1.0 - sim_mask * weight) + current_frame * sim_mask * weight
                out_frame = previous_frame * (1.0 - weight) + current_frame * weight

                processed_frames[idx] = out_frame.astype(np.uint8)


        for j in range(duplication_factor):
            weight = np.sqrt(j / transition_frames)
            current_frame = current_frame * (1.0 - weight) + current_frame_sharp * weight
            processed_frames.append(current_frame.astype(np.uint8))

    # padding
    current_frame = processed_frames[0].astype(np.float32)
    for j in range(transition_frames):
        idx = -(transition_frames - j)
        weight = (j / transition_frames)
        temp_frame = processed_frames[idx].astype(np.float32) * (1.0 - weight) + current_frame * weight
        processed_frames[idx] = temp_frame.astype(np.uint8)

    write_to_video(processed_frames, vid_path)


if __name__ == "__main__":

    path_to_frames = ""
    vid_path = ""

    run_main(
        path_to_frames, 
        vid_path, 
        transition_frames=6, 
        duplication_factor=5,
        trippy=False
    )
