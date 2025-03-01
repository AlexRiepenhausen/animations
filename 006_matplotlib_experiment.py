import cv2
import dataclasses
import numpy as np
from io import BytesIO
from PIL import Image
from sys import platform
import matplotlib.pyplot as plt
import matplotlib as mpl
import japanize_matplotlib
from tqdm import tqdm
from enum import Enum
from typing import Tuple, List

from make_top_to_bottom_animation import TopToBottom
from utils import Utils


plt.rcParams['figure.figsize'] = (19.20, 10.80)
if "win" in platform or "Win" in platform:
    plt.rcParams['font.family'] = 'MS Gothic'


class Line(Enum):
    UPPER = 1
    RIGHT = 2
    LOWER = 3
    LEFT = 4


@dataclasses.dataclass
class PlotParameters:
    line_color: str = "blue"
    line_style: str = "-"
    line_width: float = 1.0
    readjust_y_lims: bool = False
    min_x: float = 0.0
    max_x: float = 1.0
    rng_x: float = 1.0
    min_y: float = 0.0
    max_y: float = 1.0
    rng_y: float = 1.0
    title: str = ""


def simulate_data(num_items=1000):
    xs = np.arange(0, num_items).astype(np.float32)
    ys = np.random.normal(0.0, 1.0, num_items)
    ys = np.cumsum(ys)
    return xs, ys


def write_to_video(processed_frames, vid_path):
    dims = (processed_frames[0].shape[1], processed_frames[0].shape[0])
    num_frames = len(processed_frames)
    out = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, dims)
    for i in tqdm(range(num_frames)):
        out.write(processed_frames[i])  
    out.release()


def generate_single_frame(
    xs: np.ndarray, 
    ys: np.ndarray, 
    current_x: np.ndarray, 
    current_y: np.ndarray, 
    plot_params: PlotParameters
) -> np.ndarray:

    plt.plot(
        xs, 
        ys, 
        color=plot_params.line_color, 
        linestyle=plot_params.line_style, 
        linewidth=plot_params.line_width, 
        alpha=1.00
    )

    plt.ylim(plot_params.min_y, plot_params.max_y)
    plt.xlim(plot_params.min_x, plot_params.max_x)
    plt.title(plot_params.title, fontsize=20)

    x_step = plot_params.rng_x / 7
    y_step = plot_params.rng_y / 7

    x_tick_items = np.arange(plot_params.min_x + x_step, plot_params.max_x, x_step)
    y_tick_items = np.arange(plot_params.min_y + y_step, plot_params.max_y, y_step)

    x_tick_items = [round(item, 2) for item in x_tick_items]
    y_tick_items = [round(item, 2) for item in y_tick_items]

    plt.xticks(x_tick_items, x_tick_items, fontsize=14)
    plt.yticks(y_tick_items, y_tick_items, fontsize=14)

    for y in y_tick_items:
        plt.hlines(
            y, 
            xmin=plot_params.min_x, 
            xmax=current_x, 
            lw=0.75, 
            color="silver",
            linestyle="--"
        )

    for x in x_tick_items:
        if x <= current_x:
            plt.vlines(
                x,
                ymin=current_y,
                ymax=plot_params.max_y,
                lw=0.75,
                color="silver",
                linestyle="--"
            )

    buf = BytesIO()
    plt.savefig(buf, format='jpg')
    buf.seek(0)

    tmp_buf = Image.open(buf)
    frame = np.array(tmp_buf)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = cv2.filter2D(frame, -1, Utils.KRNL02)

    buf.close()
    plt.close()

    return frame


def draw_single_line(
    line_portion: Line,
    video_frames: list,
    canvas_frame: np.ndarray,
    reference_frame: np.ndarray,
    upper_y: int,
    lower_y: int,
    left_x: int,
    right_x: int,
    num_frames: int=20
) -> Tuple[List, np.ndarray]:

    for i in range(num_frames):

        if line_portion is Line.UPPER:
            num_items = right_x - left_x
            idx = left_x + int(num_items * (i / (num_frames - 1)))
            new_frame = np.copy(canvas_frame)
            new_frame[upper_y-1:upper_y+1, left_x:idx] = reference_frame[upper_y-1:upper_y+1, left_x:idx]
            video_frames.append(new_frame)

        if line_portion is Line.RIGHT:
            num_items = lower_y - upper_y
            idx = upper_y + int(num_items * (i / (num_frames - 1)))
            new_frame = np.copy(canvas_frame)
            new_frame[upper_y:idx, right_x-1:right_x+1] = reference_frame[upper_y:idx, right_x-1:right_x+1]
            video_frames.append(new_frame)

        if line_portion is Line.LOWER:
            num_items = right_x - left_x
            idx = right_x - int(num_items * (i / (num_frames - 1)))
            new_frame = np.copy(canvas_frame)
            new_frame[lower_y-1:lower_y+1, idx:right_x] = reference_frame[lower_y-1:lower_y+1, idx:right_x]
            video_frames.append(new_frame)

        if line_portion is Line.LEFT:
            num_items = lower_y - upper_y
            idx = lower_y - int(num_items * (i / (num_frames - 1)))
            new_frame = np.copy(canvas_frame)
            new_frame[idx:lower_y, left_x-1:left_x+1] = reference_frame[idx:lower_y, left_x-1:left_x+1]
            video_frames.append(new_frame)

    canvas_frame = np.copy(video_frames[-1])
    return video_frames, canvas_frame


def init_initial_frames(start_frame):

    y_axis_frame = np.copy(start_frame)
    y_axis_frame[:,239:] = (255, 255, 255)

    x_axis_frame = np.copy(start_frame)
    x_axis_frame[:963,:] = (255, 255, 255)

    box_frame = np.copy(start_frame)
    box_frame[963:,:] = (255, 255, 255)
    box_frame[:,:239] = (255, 255, 255)

    # sort into four spokes
    ys, xs, _ = np.where(box_frame < 100)
    unique_ys, unique_y_counts = np.unique(ys, return_counts=True)
    left_y_idx, right_y_idx = np.argsort(unique_y_counts)[::-1][:2]
    upper_y, lower_y = unique_ys[left_y_idx], unique_ys[right_y_idx]

    unique_xs, unique_x_counts = np.unique(xs, return_counts=True)
    upper_x_idx, lower_x_idx = np.argsort(unique_x_counts)[::-1][:2]
    right_x, left_x = unique_xs[upper_x_idx], unique_xs[lower_x_idx]

    video_frames = []
    canvas_frame = np.full(start_frame.shape, 255).astype(np.uint8)

    for line_direction in [Line.UPPER, Line.RIGHT, Line.LOWER, Line.LEFT]:
        video_frames, canvas_frame = draw_single_line(
            line_direction,
            video_frames,
            canvas_frame,
            box_frame,
            upper_y,
            lower_y,
            left_x,
            right_x,
            num_frames=15
        )

    ttb = TopToBottom(
        kernel_size=12, 
        padding_front_color=(255, 255, 255),
        padding_frames_front=0,
        padding_frames_back=4,
        frame_repetition=1
    )

    y_axis_frames = ttb.draw_to_array(y_axis_frame, np.full(y_axis_frame.shape, 255))
    for frame in y_axis_frames:
        frame[:,239:] = canvas_frame[:,239:]
        video_frames.append(frame)
    canvas_frame = np.copy(video_frames[-1])

    x_axis_frames = ttb.draw_to_array(x_axis_frame, np.full(y_axis_frame.shape, 255))
    for frame in x_axis_frames:
        frame[:963,:] = canvas_frame[:963,:]
        video_frames.append(frame)

    return video_frames


def run_main(
    x_padding: np.float32 = 0.03,
    y_padding: np.float32 = 0.03,
    seconds: np.float32 = 5,
    line_color: str = "green",
    line_style: str = "--",
    line_width: np.float32 = 1.0,
    padding_frames_back: np.int32 = 30,
    readjust_y_lims: bool = False,
    title: str = "",
    vid_path: str = "C:/Users/alexe/OneDrive/Desktop/test/test.mp4"
):

    xs, ys = simulate_data(num_items=1000)
    ys_sub = np.full(xs.shape[0], np.nan)

    rng_x = np.max(xs) - np.min(xs)
    min_x = np.min(xs) - rng_x * x_padding
    max_x = np.max(xs) + rng_x * x_padding

    rng_y = np.max(ys) - np.min(ys)
    min_y = np.min(ys) - rng_y * y_padding
    max_y = np.max(ys) + rng_y * y_padding    

    plot_params = PlotParameters()
    plot_params.line_color = line_color
    plot_params.line_style = line_style
    plot_params.line_width = line_width
    plot_params.readjust_y_lims = readjust_y_lims
    plot_params.min_x = min_x
    plot_params.max_x = max_x
    plot_params.rng_x = max_x - min_x
    plot_params.min_y = min_y
    plot_params.max_y = max_y
    plot_params.rng_y = max_y - min_y
    plot_params.title = title

    start_frame = generate_single_frame(
        xs,
        ys_sub,
        plot_params.min_x,
        plot_params.max_y,
        plot_params
    )

    video_frames = init_initial_frames(start_frame)

    current_x, current_y = 0.0, 0.0
    num_items, num_frames = xs.shape[0], seconds * 30

    for i in tqdm(range(num_frames)):

        idx = int(num_items * i / num_frames)
        progress = 1.0 if 4.0 * (i / num_frames) >= 1.0 else 4.0 * (i / num_frames)
        current_x = plot_params.max_x if 4.0 * xs[idx] >= plot_params.max_x else 4.0 * xs[idx]
        current_y = plot_params.max_y - plot_params.rng_y * progress

        ys_sub[:idx] = ys[:idx]
        frame = generate_single_frame(xs, ys_sub, current_x, current_y, plot_params)
        video_frames.append(frame)

    end_frame = generate_single_frame(xs, ys, plot_params.max_x, plot_params.min_y, plot_params)
    video_frames.append(end_frame)

    for _ in range(padding_frames_back):
        video_frames.append(end_frame)

    write_to_video(video_frames, vid_path)


if __name__ == "__main__":
    run_main(
        x_padding=0.03,
        y_padding=0.03,
        seconds=5,
        line_color="blue",
        line_style="-",
        line_width=1.0,
        padding_frames_back=30,
        readjust_y_lims=False,
        title="TEST",
        vid_path=""
    )
