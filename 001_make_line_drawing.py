import cv2
import numpy as np
from tqdm import tqdm
from typing import Union


def init_iteration_index_pairs(num_pixels, num_frames):

    step_size = num_pixels // num_frames

    iteration_idxs = [i for i in range(0, num_pixels, step_size)]
    if iteration_idxs[-1] < num_pixels - 1:
        iteration_idxs += [num_pixels - 1]

    index_pairs = []
    for i in range(1, len(iteration_idxs)):
        idx0, idx1 = iteration_idxs[i - 1], iteration_idxs[i]
        index_pairs.append((idx0, idx1))

    return index_pairs


def sort_pixels_by_connectivity(ys, xs, max_dist=6):

    num_items = len(ys)
    idxs = np.arange(0, num_items).astype(np.int32)

    candidate_dict = {}

    for idx in tqdm(idxs):

        y, x = ys[idx], xs[idx]
        y_idxs = set(np.where(np.abs(ys - y) < max_dist)[0])
        x_idxs = set(np.where(np.abs(xs - x) < max_dist)[0])

        candidate_idxs = y_idxs.intersection(x_idxs)
        candidate_dict[idx] = candidate_idxs
   
    path_idxs = [0]
    already_used_idxs = set()
    already_used_idxs.add(0)
    main_idx = 0

    while len(already_used_idxs) < len(candidate_dict):

        next_idxs, already_used_idxs = add_new_segment_idxs(path_idxs, already_used_idxs, candidate_dict, main_idx)
        num_next_idxs = len(next_idxs)

        if num_next_idxs > 0:
            for i in range(num_next_idxs):
                already_used_idxs.add(next_idxs[i])
                path_idxs.append(next_idxs[i])
            main_idx = next_idxs[0]
        else:

            for main_idx in path_idxs:

                next_idxs, already_used_idxs = add_new_segment_idxs(path_idxs, already_used_idxs, candidate_dict, main_idx)
                num_next_idxs = len(next_idxs)

                if num_next_idxs > 0:
                    for i in range(num_next_idxs):
                        already_used_idxs.add(next_idxs[i])
                        path_idxs.append(next_idxs[i])
                    main_idx = next_idxs[0]
                    break

    ys = ys[path_idxs]
    xs = xs[path_idxs]

    return ys, xs


def add_new_segment_idxs(path_idxs, already_used_idxs, candidate_dict, main_idx):
    new_segment_idxs = []
    already_used_idxs.add(main_idx)
    for cnd_idx in candidate_dict[main_idx]:
        if cnd_idx not in already_used_idxs:
            new_segment_idxs.append(cnd_idx)
    return new_segment_idxs, already_used_idxs


def generate_animation_frames(
    mask_img: np.ndarray, 
    num_frames: int, 
    bkgrnd_img: np.ndarray, 
    additional_obj_img: np.ndarray,
    reverse_dir: bool,
    background_color: int
) -> list:

    gray_scale = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray_scale != background_color)
    ys, xs = sort_pixels_by_connectivity(ys, xs, max_dist=12)

    if reverse_dir:
        ys, xs = ys[::-1], xs[::-1]

    num_pixels = len(xs)
    index_pairs = init_iteration_index_pairs(num_pixels, num_frames)

    splitted_images = []
    for idx0, idx1 in index_pairs:
        y_idx, x_idx = ys[idx0:idx1], xs[idx0:idx1]
        bkgrnd_img[y_idx, x_idx] = mask_img[y_idx, x_idx]
        splitted_images.append(bkgrnd_img.astype(np.uint8))

    splitted_images = add_smoothing_effect_to_img_sequence(splitted_images)

    if additional_obj_img is not None:

        dim_y, dim_x = additional_obj_img.shape[0], additional_obj_img.shape[1]
        obj_mask = np.ones(additional_obj_img.shape)
        additional_obj_img_gray = cv2.cvtColor(additional_obj_img, cv2.COLOR_BGR2GRAY)
        idxs = np.where(additional_obj_img_gray >= background_color - 2)
        obj_mask[idxs] = 0.0

        count = 0
        for idx0, idx1 in tqdm(index_pairs):

            # adjust indices (stabilization)
            y_idx = np.mean(ys[idx0:idx1][:10]).astype(np.int32)
            x_idx = np.mean(xs[idx0:idx1][:10]).astype(np.int32)

            y_idx = int(ys[idx1] * 0.5 + y_idx * 0.5)
            x_idx = int(xs[idx1] * 0.5 + x_idx * 0.5)

            y0, y1 = y_idx - dim_y, y_idx
            x0, x1 = x_idx, x_idx + dim_x

            current_img = splitted_images[count].astype(np.float32)
            current_img[y0:y1, x0:x1] = current_img[y0:y1, x0:x1] * (1.0 - obj_mask) + additional_obj_img * obj_mask

            splitted_images[count] = current_img.astype(np.uint8)
            count += 1

    return splitted_images


def write_to_video(img, splitted_images, vid_path):
    num_frames = len(splitted_images)
    out = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (img.shape[1], img.shape[0]))
    for i in tqdm(range(num_frames)):
        out.write(splitted_images[i])  
    out.release()


def add_smoothing_effect_to_img_sequence(splitted_images):
    for i in tqdm(range(1, len(splitted_images))):
        merged = 0.5 * splitted_images[i - 1] + 0.5 * splitted_images[i]
        splitted_images[i] = merged.astype(np.uint8)
    return splitted_images


def run_main(
    vid_path: str,
    mask_path: str,
    background_img_path: Union[str, int] = 255,  
    additional_obj_path: Union[str, None] = None, 
    num_frames:int=90, 
    padding_frames_front:int=1,
    padding_frames_back:int=90,
    reverse_dir: bool = False,
):

    mask_img = cv2.imread(mask_path).astype(np.float32)

    bkgrnd_img = None
    if isinstance(background_img_path, str):
        bkgrnd_img = cv2.imread(background_img_path).astype(np.float32)
    elif isinstance(background_img_path, int):
        bkgrnd_img = np.full(mask_img.shape, background_img_path).astype(np.float32)
    else:
        print("Background img path has to be string or int")
        exit(0)

    additional_obj_img = None
    if additional_obj_path is not None:
        additional_obj_img = cv2.imread(additional_obj_path).astype(np.float32)

    splitted_images = generate_animation_frames(
        mask_img, 
        num_frames, 
        bkgrnd_img, 
        additional_obj_img, 
        reverse_dir,
        background_color=255
    )

    splitted_images = [bkgrnd_img.astype(np.uint8)] * padding_frames_front + splitted_images + [mask_img.astype(np.uint8)] * padding_frames_back

    write_to_video(mask_img, splitted_images, vid_path)


if __name__ == "__main__":
    run_main(
        vid_path = "",
        mask_path = "",
        background_img_path = 255,
        additional_obj_path = None,
        num_frames=60,
        padding_frames_front=1,
        padding_frames_back=30,
        reverse_dir = True
    )
