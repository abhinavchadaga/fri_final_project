from elevator_model import model
import ocr
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import pytesseract

from mrcnn import visualize

path_to_image = './elevator_panels/val/2.jpg'
img = cv2.imread(path_to_image)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.asarray(img)

results = model.detect([img], verbose=1)
r = results[0]

class_names = ['BG', 'label', 'button']

visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])
label_class = 1
button_class = 2
masks = r["masks"]
masks = masks.astype(np.uint8)
print(masks[0].shape)
class_ids = r["class_ids"]
label_bounding_boxes = []
for i, box in enumerate(r['rois']):
    if class_ids[i] == label_class:
        label_bounding_boxes.append(box)
print(label_bounding_boxes)

label_masks = masks[:, :, np.where(class_ids == label_class)[0]]
labels = []
for i in range(label_masks.shape[2]):
    temp = skimage.io.imread(path_to_image)
    box = label_bounding_boxes[i]
    for j in range(temp.shape[2]):
        temp[:, :, j] = temp[:, :, j] * label_masks[:, :, i]
    temp = temp[box[0]:box[2], box[1]:box[3]]
    labels.append(temp)
    plt.figure(figsize=(8, 8))
    plt.imshow(temp)

plt.show()

cv2.cvtColor(labels[2], cv2.COLOR_BGR2GRAY)

(T, preprocessedImage) = cv2.threshold(
    labels[3], 70, 255, cv2.THRESH_BINARY)
preprocessedImage = cv2.bitwise_not(preprocessedImage)

cv2.imshow("window", preprocessedImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(pytesseract.image_to_string(preprocessedImage))
