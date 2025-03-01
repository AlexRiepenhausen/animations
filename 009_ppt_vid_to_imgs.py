import cv2
import numpy as np


def run_main(
    vid_file_path: str,
    slides_dir: str
) -> None:
    
    cap = cv2.VideoCapture(vid_file_path)
    if (cap.isOpened() == False):
        print("Error opening video file")

    prev_frame, slide_count = None, 0
    while(cap.isOpened()):

        ret, frame = cap.read()

        if ret == True:

            frame = cv2.resize(frame, (1920, 1080), cv2.INTER_CUBIC)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

            if slide_count == 0:

                cv2.imwrite(slides_dir + f"slide_{slide_count:04d}.png", frame.astype(np.uint8))
                print(f"Slide {slide_count:04d} saved")
                prev_frame = gray_frame
                slide_count += 1

            else:

                diffs_raw = np.abs(gray_frame - prev_frame)
                diffs_raw[diffs_raw < 10] = 0

                if np.max(diffs_raw) > 0:
                    diffs = 255 * diffs_raw / np.max(diffs_raw)
                    diffs = diffs.astype(np.uint8)
                    diffs = cv2.medianBlur(diffs, 3)
                    idxs = np.where(diffs == 0)
                    diffs_raw[idxs] = 0

                if np.sum(diffs_raw) > 8000:
                    print(np.sum(diffs_raw))
                    cv2.imwrite(slides_dir + f"slide_{slide_count:04d}.png", frame.astype(np.uint8))
                    print(f"Slide {slide_count} saved")
                    slide_count += 1

                prev_frame = gray_frame

        else:
            break

    cap.release()


if __name__ == "__main__":
    run_main(
        vid_file_path="",
        slides_dir=""
    )
