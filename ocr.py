import cv2

import pytesseract

# https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/doc/doc_en/training_en.md

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
        preprocessed_image, config=config_str, lang='my-custom-model', output_type=pytesseract.Output.DICT)

    # try:
    #     cv2.imshow("Test", preprocessed_image)
    #     cv2.waitKey(0)
    # except:
    #     print("Next")

    # print("\n\n" + str(button_image_boxes['text']) + "\n")
    for i in range(len(button_image_boxes['text'])):
        if button_image_boxes['text'][i] != '' and int(button_image_boxes['conf'][i]) >= conf:
            print(str(button_image_boxes['text']))
            x, y, w, h = 0, 0, 0, 0
            (x, y, w, h) = (button_image_boxes['left'][i], button_image_boxes['top'][i], button_image_boxes['width'][i], button_image_boxes['height'][i])
            if minw < w < maxw and minh < h < maxh:
                # label = "Not Recognized"
                # if button_image_boxes['text'][i] in symbolsToWords:
                #     label = symbolsToWords.get(button_image_boxes['text'][i])
                
                label = button_image_boxes['text'][i]

                cv2.rectangle(preprocessed_image, pt1=(x, y), pt2=(
                    x + w, y + h), color=text_clr, thickness=stroke_width)
                cv2.putText(img=preprocessed_image, text=label, org=(
                    x + 5, y + h - 10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1,
                            color=text_clr, thickness=1)
                print(label + " :: " + str(button_image_boxes['conf'][i]) + " -- " + str(
                    x) + ":" + str(y) + ":" + str(w) + ":" + str(h))

                return preprocessed_image, (x, y, w, h), label


# Replace buttonMat with cv2 image file and call this function

#   0    Orientation and script detection (OSD) only.
#   1    Automatic page segmentation with OSD.
#   2    Automatic page segmentation, but no OSD, or OCR. (not implemented)
#   3    Fully automatic page segmentation, but no OSD. (Default)
#   4    Assume a single column of text of variable sizes.
#   5    Assume a single uniform block of vertically aligned text.
#   6    Assume a single uniform block of text.
#   7    Treat the image as a single text line.
#   8    Treat the image as a single word.
#   9    Treat the image as a single word in a circle.
#  10    Treat the image as a single character.
#  11    Sparse text. Find as much text as possible in no particular order.
#  12    Sparse text with OSD.
#  13    Raw line. Treat the image as a single text line,
#        bypassing hacks that are Tesseract-specific.


def button_ocr(button_mat):
    scale = 1
    new_button_mat = cv2.resize(button_mat, (int(button_mat.shape[1] * scale), int(button_mat.shape[0] * scale)), interpolation = cv2.INTER_AREA)

    # cv2.imshow("Test1", button_mat)
    # print(button_mat.shape)
    # cv2.imshow("Test2", new_button_mat)
    # print(new_button_mat.shape)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return find_boxes(new_button_mat, (0, 0, 255), 0, 1, int(button_mat.shape[0] * scale), 1, int(button_mat.shape[1] * scale), "--psm 8 --dpi 70", 5)

#make training MODEL_NAME=my-custom-model START_MODEL=eng TESSDATA=./data/my-custom-model-ground-truth MAX_ITERATIONS=100000 TARGET_ERROR_RATE=0.01 WEIGHT_RANGE=0.1
#sudo mv ./data/my-custom-model.traineddata /usr/share/tesseract-ocr/4.00/tessdata
#sudo tesseract --psm 8 -l my-custom-model ../Documents/fri_final_project/0.png -

