import re
import json
import wave
import pyaudio
import numpy as np

from tqdm import tqdm
from vosk import Model, KaldiRecognizer

from .text_parser import TextParser


class AudioTimeStamps:

    def __init__(
        self,
        mecab_path,
        script_path,
        audio_path,
        model_path
    ):

        self.txt_parser = TextParser(mecab_path)
        self.subtitle_lines = self._read_subtitles(script_path)
        self.wf, self.framerate = self._open_audio_file(audio_path)
        self.model = Model(model_path)

    def map_time_stamps_to_subtitles(self):

        new_words, new_romajis, starts, ends = self._audio_to_text()
        original_romaji_subtitles = self._subtitle_lines_to_romaji()
        start_indices = self._init_start_indices_from_similarity_scores(original_romaji_subtitles, new_romajis)
        start_indices = self._remove_backward_moving_indices(start_indices)
        start_indices = self._readjust_indices(original_romaji_subtitles, new_romajis, start_indices)
        start_indices = self._remove_backward_moving_indices(start_indices)
        index_pairs = self._convert_indices_to_pairs(start_indices)

        timestamps = [(starts[idx0], starts[idx1]) for idx0, idx1 in index_pairs]
        timestamps[-1] = (timestamps[-1][0], timestamps[-1][0] * 1.02)

        for i, (idx0, idx1) in enumerate(index_pairs):

            warning = "( )"
            subtitle_start = original_romaji_subtitles[i].split()[0]
            if subtitle_start != new_romajis[idx0]:
                warning = "(!)"

            print(warning, f"{i:04d}", self.subtitle_lines[i], "|", new_words[idx0], "|", timestamps[i][0], "|", timestamps[i][1])

            if timestamps[i][1] < timestamps[i][0]:
                print("Timestamps mismatch. Stop")
                exit(0)

        return self.subtitle_lines, timestamps

    def _convert_indices_to_pairs(self, start_indices):
        index_pairs = []
        for i in range(1, len(start_indices)):
            idx0, idx1 = start_indices[i - 1], start_indices[i]
            index_pairs.append((idx0, idx1))
        index_pairs.append((start_indices[-1], start_indices[-1]))
        return index_pairs

    def _init_start_indices_from_similarity_scores(self, original_romaji_subtitles, new_romajis):

        start_indices = []
        num_subtitles = len(original_romaji_subtitles)
        num_romajis = len(new_romajis)

        for i in tqdm(range(num_subtitles)):

            current_subtitle_sub_words = original_romaji_subtitles[i].split()
            num_sub_words = len(current_subtitle_sub_words)

            distances = []
            for j in range(num_romajis):

                start = j
                end = j + num_sub_words if j + num_sub_words <= num_romajis else num_romajis

                score = 0
                for k in range(start, end):
                    if new_romajis[k] in current_subtitle_sub_words:
                        score += 1

                distances.append(score)

            start_idx = np.argmax(distances)
            start_indices.append(start_idx)

        start_indices = np.array(start_indices)

        return start_indices

    def _remove_backward_moving_indices(self, start_indices):

        differences = np.diff(start_indices, prepend=start_indices[0])
        idxs = np.where(differences < 0.0)[0]

        while len(idxs) > 0:

            idxs = np.where(differences < 0.0)[0]
            fixed_start_indices = np.copy(start_indices)

            for idx in idxs:
                fixed_start_indices[idx] = (start_indices[idx + 1] + start_indices[idx - 1]) // 2
            start_indices = fixed_start_indices

            differences = np.diff(start_indices, prepend=start_indices[0])
            idxs = np.where(differences < 0.0)[0]

        return start_indices

    def _readjust_indices(self, original_romaji_subtitles, new_romajis, start_indices):

        readjusted_start_idxs = []
        num_romajis = len(new_romajis)

        for i, idx in enumerate(start_indices):

            sub_ttl_start_word = original_romaji_subtitles[i].split()[0]
            start_romaji = new_romajis[idx]
            new_start_idx = idx

            if sub_ttl_start_word != start_romaji:

                candidate_idx_left = idx - 5 if idx - 5 > 0 else 0
                candidate_idx_right = idx + 6 if idx + 6 <= num_romajis else num_romajis
                distance_from_center = 7

                for candidate_idx in range(candidate_idx_left, candidate_idx_right):
                    if sub_ttl_start_word == new_romajis[candidate_idx]:
                        new_dist_from_center = np.abs(candidate_idx - idx)
                        if new_dist_from_center < distance_from_center:
                            distance_from_center = new_dist_from_center
                            start_romaji = new_romajis[candidate_idx]
                            new_start_idx = candidate_idx

            readjusted_start_idxs.append(new_start_idx)

        return readjusted_start_idxs

    def _audio_to_text(self):

        result_arr = self._get_results_from_model()
        words, romajis, starts, ends = [], [], [], []

        for i, item in enumerate(result_arr):

            start, end = item["start"], item["end"]
            conf, word = item["conf"], item["word"]
            romaji = self.txt_parser.get_romaji_string(word)

            words.append(word)
            romajis.append(romaji)
            starts.append(start)
            ends.append(end)

        return words, romajis, starts, ends

    def _get_results_from_model(self):

        rec = KaldiRecognizer(self.model, self.framerate) 
        rec.SetWords(True)

        result_arr = []
        while True:

            data = self.wf.readframes(self.framerate * 30)
            if len(data) == 0:
                break

            if not rec.AcceptWaveform(data):
                result = rec.Result()
            else:
                result = rec.Result()

            result_dict = json.loads(result)
            if "result" in result_dict:
                result_arr += result_dict["result"]

        return result_arr

    def _read_subtitles(self, script_path):
        subtitle_lines = []
        with open(script_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.replace("\n", "")
                subtitle_lines.append(line)
        return subtitle_lines

    def _subtitle_lines_to_romaji(self):
        romaji_subtitles = []
        num_sub_ttl_lines = len(self.subtitle_lines)
        for i in range(num_sub_ttl_lines):
            sub_ttl_line = self.subtitle_lines[i]
            romaji_chunk = self.txt_parser.get_romaji_string(sub_ttl_line)
            romaji_subtitles.append(romaji_chunk)
        return romaji_subtitles

    def _open_audio_file(self, audio_path):
        wf = wave.open(audio_path, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            print("Audio file must be WAV format mono PCM.")
            exit(1)
        framerate = wf.getframerate()
        return wf, framerate