import cv2

import pytesseract
import matplotlib.pyplot as plt

# Location of tesseract (must pip install pytesseract and sudo apt install tesseract-ocr)
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

symbolsToWords = {
    "M": "Mezzanine",
    "2": "Floor 2",
    "a": "Floor 1", ")": "Floor 1", "*": "Floor 1",
    "D": "Door Hold",
    "<": "Open Door", "Q": "Open Door",
    "n": "Close Door", ">": "Close Door", "®": "Close Door",
    "S": "Phone", "G": "Phone", "0": "Phone", "@": "Phone",
    "©": "Bell", "4": "Bell", "A": "Bell"
}


def findBoxes(preprocessedImage, textClr, conf, minw, maxw, minh, maxh, configStr, strokeWidth):
    buttonImageBoxes = pytesseract.image_to_data(
        preprocessedImage, config=configStr, lang='eng', output_type=pytesseract.Output.DICT)

    print(buttonImageBoxes['text'])

    for i in range(len(buttonImageBoxes['text'])):
        if buttonImageBoxes['text'] != '' and int(buttonImageBoxes['conf'][i]) > conf:
            x, y, w, h = 0, 0, 0, 0
            (x, y, w, h) = (buttonImageBoxes['left'][i], buttonImageBoxes['top']
                            [i], buttonImageBoxes['width'][i], buttonImageBoxes['height'][i])
            if minw < w and w < maxw and minh < h and h < maxh:

                label = "Not Recognized"
                if buttonImageBoxes['text'][i][0] in symbolsToWords:
                    label = symbolsToWords.get(buttonImageBoxes['text'][i][0])

                cv2.rectangle(preprocessedImage, pt1=(x, y), pt2=(
                    x+w, y+h), color=textClr, thickness=strokeWidth)
                cv2.putText(img=preprocessedImage, text=label, org=(
                    x + 5, y+h-10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=textClr, thickness=1)
                print(label + " :: " + str(buttonImageBoxes['conf'][i]) + " -- " + str(
                    x) + ":" + str(y) + ":" + str(w) + ":" + str(h))

                return preprocessedImage, (x, y, w, h), label

# Replace buttonMat with cv2 image file and call this function


def buttonOCR(buttonMat):
    (T, preprocessedImage) = cv2.threshold(
        buttonMat, 70, 255, cv2.THRESH_BINARY)
    preprocessedImage = cv2.bitwise_not(preprocessedImage)
    preprocessedImage = cv2.medianBlur(preprocessedImage, 5)
    preprocessedImage = cv2.GaussianBlur(preprocessedImage,(5,5),0)
    cv2.cvtColor(preprocessedImage, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("preprocessed image", preprocessedImage)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return findBoxes(preprocessedImage, (0, 0, 255), 10, 1, 500, 1, 500, "--psm 9", 5)

