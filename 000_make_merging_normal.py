import cv2
import numpy as np
from tqdm import tqdm

from utils import Utils


def get_file_names(path_to_frames, name, num_items):
    file_names = []
    for i in range(0, num_items + 1):
        file_name = path_to_frames + f"{name}{i:03d}.png"
        file_names.append(file_name)
    return file_names


def get_frames(file_names):
    frames = []
    for img_path in file_names:
        img = cv2.imread(img_path).astype(np.uint8)
        frames.append(img)
    return frames


def write_to_video(splitted_images, vid_path):
    img = splitted_images[0]
    num_frames = len(splitted_images)
    out = cv2.VideoWriter(
        vid_path, 
        cv2.VideoWriter_fourcc(*'mp4v'), 
        30, 
        (img.shape[1], img.shape[0])
    )
    for i in tqdm(range(num_frames)):
        out.write(splitted_images[i])  
    out.release()


def run_main(path_to_frames, name, num_items, vid_path, transition_frames=12, padding_frames=30):

    file_names = get_file_names(path_to_frames, name, num_items)
    input_frames = get_frames(file_names)

    processed_frames = [input_frames[0] for _ in range(transition_frames)]
    blank_img = np.copy(input_frames[0])

    for i in tqdm(range(1, len(input_frames))):

        current_frame = input_frames[i].astype(np.float32)
        previous_frame = input_frames[i - 1].astype(np.float32)
        idxs = np.where(current_frame < 255)

        for j in range(transition_frames):

            weight = (j / transition_frames)
            blank_img[idxs] = previous_frame[idxs] * (1.0 - weight) + current_frame[idxs] * weight
            processed_frames.append(blank_img.astype(np.uint8))

    last_frame = processed_frames[-1]
    for i in range(padding_frames):
        processed_frames.append(last_frame)

    write_to_video(processed_frames, vid_path)


if __name__ == "__main__":

    path_to_frames = ""
    name = ""
    num_items = 13
    vid_path = ""

    run_main(
        path_to_frames,
        name,
        num_items,
        vid_path,
        transition_frames=9,
        padding_frames=30
    )
