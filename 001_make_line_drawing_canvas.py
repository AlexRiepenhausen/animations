import cv2
import numpy as np
from tqdm import tqdm
from enum import Enum

from typing import Tuple, List

class Line(Enum):
    UPPER = 1
    RIGHT = 2
    LOWER = 3
    LEFT = 4



def draw_single_line(
    line_portion: Line,
    video_frames: list,
    canvas_frame: np.ndarray,
    reference_frame: np.ndarray,
    upper_y: int,
    lower_y: int,
    left_x: int,
    right_x: int,
    num_frames: int=20,
    lw: int=1
) -> Tuple[List, np.ndarray]:

    for i in range(num_frames):

        if line_portion is Line.UPPER:
            num_items = right_x - left_x
            idx = left_x + int(num_items * (i / (num_frames - 1)))
            new_frame = np.copy(canvas_frame)
            new_frame[upper_y-lw:upper_y+lw, left_x:idx] = reference_frame[upper_y-lw:upper_y+lw, left_x:idx]
            video_frames.append(new_frame)

        if line_portion is Line.RIGHT:
            num_items = lower_y - upper_y
            idx = upper_y + int(num_items * (i / (num_frames - 1)))
            new_frame = np.copy(canvas_frame)
            new_frame[upper_y:idx, right_x-lw:right_x+lw] = reference_frame[upper_y:idx, right_x-lw:right_x+lw]
            video_frames.append(new_frame)

        if line_portion is Line.LOWER:
            num_items = right_x - left_x
            idx = right_x - int(num_items * (i / (num_frames - 1)))
            new_frame = np.copy(canvas_frame)
            new_frame[lower_y-lw:lower_y+lw, idx:right_x] = reference_frame[lower_y-lw:lower_y+lw, idx:right_x]
            video_frames.append(new_frame)

        if line_portion is Line.LEFT:
            num_items = lower_y - upper_y
            idx = lower_y - int(num_items * (i / (num_frames - 1)))
            new_frame = np.copy(canvas_frame)
            new_frame[idx:lower_y, left_x-lw:left_x+lw] = reference_frame[idx:lower_y, left_x-lw:left_x+lw]
            video_frames.append(new_frame)

    canvas_frame = np.copy(video_frames[-1])
    return video_frames, canvas_frame


def write_to_video(processed_frames, vid_path):
    dims = (processed_frames[0].shape[1], processed_frames[0].shape[0])
    num_frames = len(processed_frames)
    out = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, dims)
    for i in tqdm(range(num_frames)):
        out.write(processed_frames[i])  
    out.release()


def run_main(
    img_path: str,
    vid_path: str,
    padding_frames_front: int = 30,
    padding_frames_back: int = 30,
    num_frames: int = 60,
    line_width: int = 5
) -> None:

    img = cv2.imread(img_path)
    ys, xs, _ = np.where(img < 255)
    upper_y = np.min(ys)
    lower_y = np.max(ys)
    left_x = np.min(xs)
    right_x = np.max(xs)

    video_frames = []
    canvas_frame = np.full(img.shape, 255).astype(np.uint8)

    for line_direction in [Line.UPPER, Line.RIGHT, Line.LOWER, Line.LEFT]:
        video_frames, canvas_frame = draw_single_line(
            line_direction,
            video_frames,
            canvas_frame,
            img,
            upper_y,
            lower_y,
            left_x,
            right_x,
            num_frames=num_frames // 4,
            lw=line_width
        )

    for _ in range(padding_frames_back):
        video_frames.append(img)

    write_to_video(video_frames, vid_path)


if __name__ == "__main__":
    run_main(
        img_path ="",
        vid_path ="",
        padding_frames_front=30,
        padding_frames_back=30,
        num_frames=60,
        line_width=4
    )
