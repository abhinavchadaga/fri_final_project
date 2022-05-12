from elevator_model import model
from mrcnn import visualize
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.io

import warnings
warnings.filterwarnings("ignore")

path_to_image = './elevator_panels/val/0.jpg'
img = cv2.imread(path_to_image)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_arr = np.asarray(img)

results = model.detect([img_arr], verbose=1)
r = results[0]

label_class = 1
button_class = 2
masks = r["masks"]
class_ids = r["class_ids"]

label_masks = masks[:, :, np.where(class_ids == label_class)[0]]
button_masks = masks[:, :, np.where(class_ids == button_class)[0]]
labels = []
buttons = []
for i in range(label_masks.shape[2]):
    label_tmp = skimage.io.imread('./elevator_panels/val/0.jpg')
    button_tmp = skimage.io.imread('./elevator_panels/val/0.jpg')
    for j in range(label_tmp.shape[2]):
        label_tmp[:, :, j] = label_tmp[:, :, j] * label_masks[:, :, i]
        button_tmp[:, :, j] = button_tmp[:, :, j] * button_masks[:, :, i]
    label_tmp = cv2.resize(label_tmp, (1024, 1024))
    button_tmp = cv2.resize(button_tmp, (1024, 1024))
    labels.append(label_tmp)
    buttons.append(button_tmp)

for i, label in enumerate(labels):
    cv2.imshow("label " + str(i), label.astype(np.uint8))
    cv2.waitKey(0)

for i, button in enumerate(buttons):
    cv2.imshow("label " + str(i), button.astype(np.uint8))
    cv2.waitKey(0)

cv2.destroyAllWindows()


# cv2.imshow("label masks", m.astype(np.uint8))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# plt.imshow(m.astype(np.uint8), cmap="Blues")
# plt.show()

# class_names = ['BG', 'label', 'button']


# visualize.display_top_masks(img_arr, r["masks"], r["class_ids"], class_names)
 