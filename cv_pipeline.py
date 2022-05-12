from elevator_model import model
import ocr
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import copy
from mrcnn import visualize

#path_to_image = './elevator_panels/val/0.jpg'
#img = cv2.imread(path_to_image)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#img = np.asarray(img)

path_to_image = './elevator_panels/val/6.jpg'
img = cv2.imread(path_to_image)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.asarray(img)

results = model.detect([img], verbose=1)
r = results[0]

class_names = ['BG', 'label', 'button']

label_class = 1
button_class = 2
masks = r["masks"]
masks = masks.astype(np.uint8)
class_ids = r["class_ids"]
label_bounding_boxes = []
for i, box in enumerate(r['rois']):
    if class_ids[i] == label_class:
        label_bounding_boxes.append(box)

label_masks = masks[:, :, np.where(class_ids == label_class)[0]]
labels = []
for i in range(label_masks.shape[2]):
    #temp = skimage.io.imread(path_to_video)
    temp = skimage.io.imread(path_to_image)
    box = label_bounding_boxes[i]
    # for j in range(temp.shape[2]):
    #     temp[:, :, j] = temp[:, :, j] * label_masks[:, :, i]
    temp = temp[box[0]-25:box[2]+25, box[1]-50:box[3]+50]
    labels.append(temp)
    # plt.figure(figsize=(8, 8))

for i in range(len(labels)):
    try:    
        output, (rectX, rectY, rectW, rectH), text = ocr.buttonOCR(labels[i])
    except:
        print("Not recognized")

    # cv2.imshow(str(i) + " window " + text, output)
    textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=2, thickness=3)
    cv2.putText(img = img, text = text, org=(int(label_bounding_boxes[i][1] + (label_bounding_boxes[i][3]-label_bounding_boxes[i][1])/2 - textSize[0][0]/2), int(label_bounding_boxes[i][2]-30)), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=2, color=(255, 255, 255), thickness=3)
    # cv2.rectangle(img, pt1=(label_bounding_boxes[i][1], label_bounding_boxes[i][0]), pt2=(label_bounding_boxes[i][3], label_bounding_boxes[i][2]), color=(255, 0, 0), thickness=5)

visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

# cv2.waitKey(1000)
# plt.close('all')

# cv2.waitKey()
# cv2.destroyAllWindows()