import re
import MeCab
import unicodedata
from .common import RomKan


class TextParser:

    def __init__(self, unidic_path):
        self.jp_tagger = MeCab.Tagger(unidic_path)
        self.punctuation = ["。", "、", "；", "：", "（", "）", ".", "「", "」", "!", ";", "？", "?"]

    def get_romaji_string(self, in_text):

        in_text = in_text.replace("、", "")
        tokenized_sample_words = self.get_tokenized_sample_words(in_text)

        romaji_items = []
        for word in tokenized_sample_words:
            romaji_w = self._convert_token_to_romaji(word)
            romaji_items.append(romaji_w)

        romaji_str = " ".join(romaji_items)
        romaji_str = romaji_str.strip()

        return romaji_str

    def get_tokenized_sample_words(self, in_text):

        def process_token(tk):
            if self._is_japanese(tk):
                return tk
            else:
                tk = tk.replace('"', '').replace("・", "")
                if tk.isalpha():
                    return tk.lower()
                return tk

        tokenized_sample_words, previous = [], ""
        node = self.jp_tagger.parseToNode(in_text)

        while node:
            tk_parsed = process_token(node.surface)
            tokenized_sample_words.append(tk_parsed)
            node = node.next
            previous = tk_parsed

        return tokenized_sample_words

    def _convert_token_to_romaji(self, word):
        if not self._is_japanese(word):
            # if word == "ｎｌｐ":
                # print("PULA")
            return word
        else:
            if self._is_kanji(word):
                try:
                    word = self._kanji_to_hiragana(word)
                except:
                    pass
            word = RomKan.to_roma(word)
            return word

    def _is_japanese(self, word):
        japanese_exists = re.search("[\u3040-\u30ff]|[\u4e00-\u9FFF]", word)
        if japanese_exists is not None:
            return True
        if word in self.punctuation:
            return True
        return False

    def _is_kanji(self, word):
        try:
            for c in word:
                char_metadata = unicodedata.name(c)
                if "CJK" in char_metadata:
                    return True
            return False
        except:
            return False

    def _kanji_to_hiragana(self, word):
        node = self.jp_tagger.parseToNode(word)
        node = node.next
        items = node.feature.split(",")
        return items[6]
