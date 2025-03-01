import cv2
import numpy as np
from tqdm import tqdm


def run_main(
    input_path="C:/Users/alexe/OneDrive/Pictures/109_FUJI/DSCF",
    out_vid_path="C:/Users/alexe/OneDrive/Pictures/timelapse0.mp4",
    img_type="jpg",
    start_idx=1,
    end_idx=2,
    framerate=30,
    output_height=2160,
    output_width=3840,
    blur_strength=0.1
):
    out, previous_frame  = None, None
    img_paths = [input_path + f"{i:04d}.{img_type}" for i in range(start_idx, end_idx + 1)]

    for img_path in tqdm(img_paths):
        frame = cv2.imread(img_path)
        if out is None:
            out = cv2.VideoWriter(
                out_vid_path, 
                cv2.VideoWriter_fourcc('M','P','4','V'), 
                framerate, 
                (output_width, output_height)
            )

        frame = cv2.resize(frame, (output_width, output_height), cv2.INTER_CUBIC)
        if previous_frame is not None:
            tmp = frame.astype(np.float32) * (1.0 - blur_strength) + previous_frame.astype(np.float32) * blur_strength
            out.write(tmp.astype(np.uint8))
            previous_frame = np.copy(tmp)
        else:
            out.write(frame.astype(np.uint8))
            previous_frame = np.copy(frame)

    out.release()


if __name__ == "__main__":
    run_main(
        input_path="",
        out_vid_path="",
        img_type="jpg",
        start_idx=1,
        end_idx=214,
        framerate=30,
        output_height=2160,
        output_width=3840,
        blur_strength=0.1
    )
