from elevator_model import model
import ocr
import cv2
import numpy as np
import skimage.io
from mrcnn import visualize


def detect(path_to_image, button_text):

    class_names = ['BG', 'label', 'button']

    img = cv2.imread(path_to_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img)

    results = model.detect([img], verbose=1)
    r = results[0]

    visualize.display_instances(
        img, r["rois"], r["masks"], r["class_ids"], class_names, r["scores"])

    label_class = 1
    button_class = 2
    masks = r["masks"]
    masks = masks.astype(np.uint8)
    class_ids = r["class_ids"]
    label_bounding_boxes = []
    for i, box in enumerate(r['rois']):
        if class_ids[i] == label_class:
            label_bounding_boxes.append(box)

    button_bounding_boxes = []
    for i, box in enumerate(r['rois']):
        if class_ids[i] == button_class:
            button_bounding_boxes.append(box)

    btn_scores = []
    for i, score in enumerate(r['rois']):
        if class_ids[i] == button_class:
            btn_scores.append(score)

    label_masks = masks[:, :, np.where(class_ids == label_class)[0]]
    button_masks = masks[:, :, np.where(class_ids == button_class)[0]]

    # visualize.display_top_masks(
    #     img, r["masks"], r["class_ids"], class_names, limit=2)

    labels = []
    for i in range(label_masks.shape[2]):
        temp = skimage.io.imread(path_to_image)
        box = label_bounding_boxes[i]
        for j in range(temp.shape[2]):
            temp[:, :, j] = temp[:, :, j] * label_masks[:, :, i]
        temp = temp[box[0]-25:box[2]+25, box[1]-50:box[3]+50]
        labels.append(temp)

    # visualize.display_images(labels)

    processed_labels = []
    label_contents = []
    for i in range(len(labels)):
        (T, preprocessedImage) = cv2.threshold(
            labels[i], 70, 255, cv2.THRESH_BINARY)
        cv2.cvtColor(preprocessedImage, cv2.COLOR_BGR2GRAY)
        preprocessedImage = np.array(preprocessedImage)
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        preprocessedImage = cv2.morphologyEx(
            preprocessedImage, cv2.MORPH_OPEN, kernel)
        preprocessedImage = cv2.bitwise_not(preprocessedImage)
        processed_labels.append(preprocessedImage)

        try:
            output, (rectX, rectY, rectW,
                     rectH), text = ocr.buttonOCR(preprocessedImage)
            label_contents.append(text)
        except:
            text = "Not Recognized"
            label_contents.append(text)

        textSize = cv2.getTextSize(
            text, fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=2, thickness=3)
        cv2.putText(img=img, text=text, org=(int(label_bounding_boxes[i][1] + (label_bounding_boxes[i][3]-label_bounding_boxes[i][1])/2 - textSize[0][0]/2), int(
            label_bounding_boxes[i][2]-30)), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=2, color=(255, 255, 255), thickness=3)

    # ocr_output = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # cv2.imwrite("./ocr_output.jpg", ocr_output)

    visualize.display_images(processed_labels)

    try:
        selected_label_bounding_box = label_bounding_boxes[label_contents.index(
            button_text)]

        y1 = selected_label_bounding_box[0]
        y2 = selected_label_bounding_box[2]
        x2 = selected_label_bounding_box[3]

        possible_buttons_idx = []

        for i, button_bounding_box in enumerate(button_bounding_boxes):
            midpoint = (button_bounding_box[0] + button_bounding_box[2]) / 2
            if midpoint > y1 and midpoint < y2 and button_bounding_box[1] > x2:
                possible_buttons_idx.append(i)

        tgt_btn = button_bounding_boxes[possible_buttons_idx[0]]
        tgt_idx = possible_buttons_idx[0]

        for button_idx in possible_buttons_idx:
            button_bounding_box = button_bounding_boxes[button_idx]
            x1 = button_bounding_box[1]
            if x1 < tgt_btn[1]:
                tgt_btn = button_bounding_box
                tgt_idx = button_idx

        # print(np.array([button_bounding_boxes[tgt_idx]]).shape)
        # print(r["rois"].shape)

        # print(np.array(button_masks).shape)
        # print(r["masks"].shape)

        # print(np.array(button_masks[:, :, tgt_idx:(tgt_idx+1)]).shape)

        visualize.display_instances(img, np.array([button_bounding_boxes[tgt_idx]]), np.array(
            button_masks[:, :, tgt_idx:(tgt_idx+1)]), np.array([button_class]), class_names, title="selecting " + button_text)

        return 1
    except:
        visualize.display_images([img])
        return 0
