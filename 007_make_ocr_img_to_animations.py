import cv2
import pytesseract
import numpy as np
from tqdm import tqdm
from pytesseract import Output


def remove_non_text_items(results_dict: dict):

    text = results_dict["text"]
    selected_idx = [True if txt != "" else False for txt in text]
    keys = list(results_dict.keys())

    for key in keys:
        items = []
        current_items = results_dict[key]
        for i, selected in enumerate(selected_idx):
            if selected:
                items.append(current_items[i])
        results_dict[key] = items

    return results_dict


def box0_encompasses_box1(box0, box1):

    x00, x01, y00, y01 = box0
    x10, x11, y10, y11 = box1

    is_same = (x10 == x00) and (x11 == x01) and (y10 == y00) and (y11 == y01)
    fits_inside = (x10 >= x00) and (x11 <= x01) and (y10 >= y00) and (y11 <= y01)

    if is_same:
        return False

    if fits_inside:
        return True

    return False


def run_filter(results_dict):

    boxes = []
    for i in range(len(results_dict['level'])):
        x0, x1 = results_dict['left'][i], results_dict['left'][i] + results_dict['width'][i]
        y0, y1 = results_dict['top'][i], results_dict['top'][i] + results_dict['height'][i]
        boxes.append((x0, x1, y0, y1))

    filtered_boxes = []
    for i, box0 in enumerate(boxes):
        num_children = np.sum([box0_encompasses_box1(box0, box1) for box1 in boxes])
        if num_children == 0:
            filtered_boxes.append(box0)

    return filtered_boxes


def generate_box_progressions(
    filtered_boxes: dict, 
    frame_adj_by_size: bool,
    y_pad: int = 0,
    x_pad: int = 0
):
    
    prev_x0 = 0
    start_flag = False
    box_progressions = {}
    x0, x1, y0, y1 = 0, 0, 0, 0

    if frame_adj_by_size:
    
        box_widths = []
        for box in filtered_boxes:
            x0, x1, _, _ = box
            box_widths.append(x1 - x0)

        max_width = max(box_widths)
        box_num_frames = np.array([round(np.log(1.0 + max_width / w)) for w in box_widths]).astype(np.int32)
        box_num_frames[box_num_frames < 1] = 1
        box_num_frames = np.abs(box_num_frames - np.max(box_num_frames) - 1)

        for i, box in enumerate(filtered_boxes):

            num_frames = box_num_frames[i]
            box_progressions[i] = {"num_frames": num_frames, "split_boxes": []}

            x0, x1, y0, y1 = box

            if x0 > prev_x0 and start_flag:
                x0 = prev_x0

            x_rng = x1 - x0
            for j in range(0, num_frames):
                progress = (j + 1) / num_frames
                right_boundary = x0 + int(progress * x_rng)
                box_progressions[i]["split_boxes"].append((x0 - x_pad, right_boundary + x_pad, y0 - y_pad, y1 + y_pad))

            prev_x0 = x1
            start_flag = True

        return box_progressions
    
    else:

        for i, box in enumerate(filtered_boxes):

            x0, x1, y0, y1 = box

            if x0 >= prev_x0 and start_flag:
                x0 = prev_x0

            box_progressions[i] = {"num_frames": 1, "split_boxes": [(x0 - x_pad, x1 + x_pad, y0 - y_pad, y1 + y_pad)]}

            prev_x0 = x1
            start_flag = True

        return box_progressions


def identify_background_color(image):
    pixels = image.reshape(-1, 3)
    colors_as_int = pixels[:, 0] * 256**2 + pixels[:, 1] * 256 + pixels[:, 2]
    unique_colors, counts = np.unique(colors_as_int, return_counts=True)
    most_frequent_index = np.argmax(counts)
    most_frequent_color_int = unique_colors[most_frequent_index]
    r = (most_frequent_color_int >> 16) & 0xFF
    g = (most_frequent_color_int >> 8) & 0xFF
    b = most_frequent_color_int & 0xFF
    return (b, g, r)


def run_main(
    img_path="./cache/ocr_test.jpg",
    vid_path="./cache/vid.mp4",
    padding_frames_front: int = 1,
    padding_frames_back: int = 90,
    y_pad: int = 0,
    x_pad: int = 0,
    language: str = "eng",
    frame_adj_by_size: bool = False
) -> None:

    img_original = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

    # mid_val = np.median(img_gray)
    # img_gray[img_gray < int(mid_val * 0.5)] = 0

    bgr_color = identify_background_color(img_original)
    img_width = img_original.shape[1]
    img_height = img_original.shape[0]

    pytesseract.pytesseract.tesseract_cmd = "C:/Users/alexe/AppData/Local/Programs/Tesseract-OCR/tesseract.exe"
    results_dict = pytesseract.image_to_data(img_gray, lang=language, output_type=Output.DICT)
    results_dict = remove_non_text_items(results_dict)

    canvas_img = np.full(img_original.shape, bgr_color).astype(np.float32)
    filtered_boxes = run_filter(results_dict)
    box_progressions = generate_box_progressions(filtered_boxes, frame_adj_by_size, y_pad, x_pad)

    out = cv2.VideoWriter(
        vid_path, 
        cv2.VideoWriter_fourcc(*'mp4v'), 
        30, 
        (img_width, img_height)
    )

    for _ in range(padding_frames_front):
        out.write(canvas_img.astype(np.uint8))
    prev_img = np.copy(canvas_img)

    for i in tqdm(box_progressions):
        for j in range(box_progressions[i]["num_frames"]):

            x0, x1, y0, y1 = box_progressions[i]["split_boxes"][j]

            canvas_img[y0:y1, x0:x1] = img_original[y0:y1, x0:x1]
            out_img = 0.5 * prev_img + 0.5 * canvas_img

            #canvas_img[y0-1:y0+1, x0:x1] = 100
            #canvas_img[y1-1:y1+1, x0:x1] = 100
            #canvas_img[y0:y1, x0-1:x0+1] = 100
            #canvas_img[y0:y1, x1-1:x1+1] = 100

            prev_img = canvas_img

            out.write(out_img.astype(np.uint8))


    for _ in range(padding_frames_back):
        out.write(img_original.astype(np.uint8))


if __name__ == "__main__":
    run_main(
        img_path="",
        vid_path="",
        padding_frames_front = 1,
        padding_frames_back = 90,
        y_pad = 0,
        x_pad = 0,
        language = "eng",
        frame_adj_by_size = True
    )
