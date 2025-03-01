import cv2
import numpy as np

from tqdm import tqdm
from PIL import ImageFont, ImageDraw, Image


class TelePrompter:

    def __init__(
        self,
        text_file_path="", 
        height=1080,
        width=1920,
        fontsize=128,
        frames_per_line=30,
    ):

        self.height = height
        self.width = width
        self.fontsize = fontsize
        self.line_width_y = fontsize + 12
        self.frames_per_line = frames_per_line
        self.line_chr_limit = int((256 * 7) / fontsize)
        self.num_lines_on_screen = int((64 * 10) / fontsize) + 1
        self.text_arr = self._read_text(text_file_path)


    def make_teleprompter_vid(self, video_path):

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (self.width, self.height))

        num_lines_total = len(self.text_arr)

        for current_line_num in tqdm(range(num_lines_total)):

            img = self._init_image(current_line_num)
            pxl_shift = self.line_width_y / self.frames_per_line

            for i in range(1, self.frames_per_line + 1):

                current_y_shift = 1 + int(pxl_shift * i)
                blank_img = np.full((self.height, self.width, 3), 255, np.uint8)
                blank_img[:-current_y_shift] = img[current_y_shift:]

                weight = i / self.frames_per_line
                padding_shift_y = 16 + self.line_width_y * self.num_lines_on_screen - current_y_shift
                blank_img[padding_shift_y:] = blank_img[padding_shift_y:] * weight + 255 * (1.0 - weight)

                out.write(blank_img.astype(np.uint8))


    def _init_image(self, current_line_num):

        img = np.full((self.height, self.width, 3), 255, np.uint8)
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)

        font = ImageFont.truetype('C:\Windows\Fonts\MSGOTHIC.TTC', self.fontsize)

        for i, text_line in enumerate(self.text_arr[current_line_num:]):
            draw.text((16, 16 + i * self.line_width_y), text_line, font=font, fill=(0,0,0,0))
            if i >= self.num_lines_on_screen:
                break

        img = np.array(img_pil)
        return img


    def _read_text(self, text_file_path):
        text_str = ""
        with open(text_file_path, "r", encoding="utf-8") as f:
            for line in f:
                text_str += line.replace("\n", "")
        text_arr = [text_str[i:i+self.line_chr_limit] for i in range(0, len(text_str), self.line_chr_limit)]
        text_arr = [""] + text_arr
        text_arr = np.array(text_arr)
        return text_arr


if __name__ == "__main__":

    teleprompt = TelePrompter(
        text_file_path="./cache/teleprompter.txt", 
        height=1080,
        width=1920,
        fontsize=128,
        frames_per_line=90
    )

    teleprompt.make_teleprompter_vid(
        video_path=""
    )
