import cv2

import pytesseract

# Location of tesseract (must pip install pytesseract and sudo apt install tesseract-ocr)
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

symbolsToWords = {
    "M": "Mezzanine",
    "a": "Floor 1", ")": "Floor 1", "*": "Floor 1", "x": "Floor 1", "x1": "Floor 1",
    "«1": "Floor 1", "*1": "Floor 1", "*|": "Floor 1", "d": "Floor 1",
    "DF": "Door Hold", "DH": "Door Hold",
    "<": "Open Door", "Q": "Open Door", "<I»": "Open Door", "4d": "Open Door", "did": "Open Door",
    "<i>": "Open Door",
    "nd": "Close Door", ">": "Close Door", "®": "Close Door", "bid": "Close Door",
    "vi": "Close Door",
    "S": "Phone", "G": "Phone", "0": "Phone", "@": "Phone", "(": "Phone",
    "©": "Bell", "A": "Bell", "4.": "Bell", "A.": "Bell",
    "1": "Floor 1",
    "2": "Floor 2",
    "3": "Floor 3",
    "4": "Floor 4",
    "5": "Floor 5",
    "6": "Floor 6",
    "7": "Floor 7",
    "8": "Floor 8",
    "9": "Floor 9",
    "10": "Floor 10",
    "11": "Floor 11",
    "12": "Floor 12",
    "13": "Floor 13",
    "14": "Floor 14",
    "15": "Floor 15",
    "16": "Floor 16",
    "17": "Floor 17",
    "18": "Floor 18",
    "19": "Floor 19",
}


def find_boxes(preprocessed_image, text_clr, conf, minw, maxw, minh, maxh, config_str, stroke_width):
    button_image_boxes = pytesseract.image_to_data(
        preprocessed_image, config=config_str, lang='eng', output_type=pytesseract.Output.DICT)

    print(button_image_boxes['text'])

    for i in range(len(button_image_boxes['text'])):
        if button_image_boxes['text'] != '' and int(button_image_boxes['conf'][i]) > conf:
            x, y, w, h = 0, 0, 0, 0
            (x, y, w, h) = (button_image_boxes['left'][i], button_image_boxes['top']
            [i], button_image_boxes['width'][i], button_image_boxes['height'][i])
            if minw < w < maxw and minh < h < maxh:
                label = "Not Recognized"
                if button_image_boxes['text'][i] in symbolsToWords:
                    label = symbolsToWords.get(button_image_boxes['text'][i])

                cv2.rectangle(preprocessed_image, pt1=(x, y), pt2=(
                    x + w, y + h), color=text_clr, thickness=stroke_width)
                cv2.putText(img=preprocessed_image, text=label, org=(
                    x + 5, y + h - 10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1,
                            color=text_clr, thickness=1)
                print(label + " :: " + str(button_image_boxes['conf'][i]) + " -- " + str(
                    x) + ":" + str(y) + ":" + str(w) + ":" + str(h))

                return preprocessed_image, (x, y, w, h), label


# Replace buttonMat with cv2 image file and call this function


def button_ocr(button_mat):
    return find_boxes(button_mat, (0, 0, 255), 0, 1, 500, 1, 500, "--psm 9", 5)
