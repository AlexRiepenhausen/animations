import cv2
import numpy as np

from tqdm import tqdm
from PIL import ImageFont, ImageDraw, Image

from subtitle_gen.audio_time_stamps import AudioTimeStamps


class SubtitleGen:

    def __init__(
        self,
        height=1080,
        width=1920,
        fontsize=32,
        framerate=29.97
    ):

        self.height = height
        self.width = width
        self.fontsize = fontsize
        self.framerate = framerate
        self.line_width_y = fontsize + 12


    def generate_subtitles(self, subtitle_lines, timestamps, video_path):

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, self.framerate, (self.width, self.height))

        # init initial blank screen
        first_sub_ttl_start = int(timestamps[0][0] * self.framerate)
        blank_img = np.full((self.height, self.width, 3), 0, np.uint8)

        for _ in range(first_sub_ttl_start):
            out.write(blank_img.astype(np.uint8))

        num_subtitle_lines = len(subtitle_lines)
        for i in tqdm(range(num_subtitle_lines)):

            sub_ttl_line = subtitle_lines[i]
            start_time, end_time = timestamps[i][0], timestamps[i][1]

            start_frame = int(start_time * self.framerate)
            end_frame = int(end_time * self.framerate)

            sub_ttl_img = self._init_image(sub_ttl_line)

            for j in range(start_frame, end_frame):
                out.write(sub_ttl_img.astype(np.uint8))


    def _init_image(self, sub_ttl_line):

        def init_pil_image():
            img = np.full((self.height, self.width, 3), 0, np.uint8)
            img_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(img_pil)
            return img_pil, draw

        # init image
        img_pil, draw = init_pil_image()

        # calculate offsets
        offset_y = self.height - self.line_width_y - 32

        # drawtext onto image
        font = ImageFont.truetype('C:\Windows\Fonts\MSGOTHIC.TTC', self.fontsize)
        draw.text((10, offset_y), sub_ttl_line, font=font, fill=(255,255,255,0))
        img = np.array(img_pil)

        # center font
        idxs = np.where(img == 255)[1]
        start_x, end_x = np.min(idxs), np.max(idxs)
        offset_x = (start_x + self.width - end_x) // 2

        img_pil, draw = init_pil_image()
        draw.text((offset_x, offset_y), sub_ttl_line, font=font, fill=(255,255,255,0))
        img = np.array(img_pil)

        return img


def run_main():

    # https://alphacephei.com/vosk/models
    # https://buddhi-ashen-dev.vercel.app/posts/offline-speech-recognition
    # pip install vosk

    ats = AudioTimeStamps(
        mecab_path="../wikienv/Lib/site-packages/unidic-lite/dicdir",
        script_path="./cache/subtitles.txt",
        audio_path="./cache/vid009_voice.wav",
        model_path="./subtitle_gen/vosk-model-ja-0.22"
    )

    subtitle_lines, timestamps = ats.map_time_stamps_to_subtitles()

    sbttlgn = SubtitleGen(
        height=1080,
        width=1920,
        fontsize=48
    )

    sbttlgn.generate_subtitles(
        subtitle_lines,
        timestamps,
        video_path=""
    )


if __name__ == "__main__":
    run_main()