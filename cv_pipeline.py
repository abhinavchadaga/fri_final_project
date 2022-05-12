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
masks = r["masks"]
class_ids = r["class_ids"]

label_masks = masks[:, :, np.where(class_ids == label_class)[0]]
labels = []
for i in range(label_masks.shape[2]):
    temp = skimage.io.imread('./elevator_panels/val/0.jpg')
    for j in range(temp.shape[2]):
        temp[:, :, j] = temp[:, :, j] * label_masks[:, :, i]
    temp = cv2.resize(temp, (1024, 1024))
    labels.append(temp)

for label in labels:
    cv2.imshow("first label", labels[0].astype(np.uint8))
    cv2.waitKey(0)

cv2.destroyAllWindows()


# cv2.imshow("label masks", m.astype(np.uint8))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# plt.imshow(m.astype(np.uint8), cmap="Blues")
# plt.show()

# class_names = ['BG', 'label', 'button']


# visualize.display_top_masks(img_arr, r["masks"], r["class_ids"], class_names)
