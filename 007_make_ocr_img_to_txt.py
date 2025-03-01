import cv2
import pytesseract
import numpy as np


def run_main(
    img_path = "./cache/ocr_test.JPG",
    black_text = False
):

    img_original = cv2.imread(img_path)
    img = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY).astype(np.float32)

    if black_text:
        idxs = np.where(img < 100)
        new_img = np.zeros(img.shape)
        new_img[idxs] = 255
        img = new_img
    else:
        idxs = np.where(img > 100)
        new_img = np.zeros(img.shape)
        new_img[idxs] = 255
        img = new_img

    img = np.dstack([img, img, img]).astype(np.uint8)
    img = cv2.dilate(img, (3, 3), iterations=1)

    pytesseract.pytesseract.tesseract_cmd = "C:/Users/alexe/AppData/Local/Programs/Tesseract-OCR/tesseract.exe"
    result = pytesseract.image_to_string(img, lang="eng")

    with open("./cache/test_txt.txt", "w", encoding="utf-8") as f:
        f.write(result)


if __name__ == "__main__":
    run_main(
        img_path = "",
        black_text = True
    )
