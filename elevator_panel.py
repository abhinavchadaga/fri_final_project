import json
import os
from re import S
import sys
import time
import numpy as np
import imgaug
import skimage.io

ROOT_DIR = os.path.abspath("./")

sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import model as modellib, utils

PATH_TO_COCO_WEIGHTS = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Extend Config Class
############################################################

class ElevatorPanelConfig(Config):
    name = "elevator_panel"

    IMAGES_PER_GPU = 2

    NUM_CLASSES = 1 + 1

    STEPS_PER_EPOCH = 100

    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class ElevatorPanelDataset(utils.Dataset):

    def load_elevator(self, dataset_dir, subset):
        # add classes
        self.add_class("elevator_panel", 1, "label")
        self.add_class("elevator_panel", 2, "button")
        self.add_class("elevator_panel", 3, "label_and_button")

        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        annotations = json.load(open(os.path.join(dataset_dir, "masks.json")))
        annotations = list(annotations.values())

        # skip images that have an empty region key
        # aka images with no annotations
        annotations = [a for a in annotations if a['regions']]

        # Add the images
        for annotation in annotations:
            # get all the shape attributes from the regions list of annotation
            # store in a list
            masks = [region["shape_attributes"] for region in annotation["regions"]]
            mask_ids = [region["region_attributes"] for region in annotation["regions"]]

            image_path = os.path.join(dataset_dir, annotation['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image("elevator_panel",
                           image_id=annotation['filename'],  # use the file name as unique img id
                           path=image_path,
                           width=width,
                           height=height,
                           masks=masks,
                           mask_ids=mask_ids)

    def load_mask(self, image_id):
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "balloon":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        m = np.zeros([info["height"], info["width"], len(info["masks"])],
                     dtype=np.uint8)

        region_info = info['region_info']
        instance_masks = []
        class_ids = []
        for i, p in enumerate(info["masks"]):
            # Get indexes of pixels inside the polygon and set them to 1
            if p["name"] == "circle":
                rr, cc = skimage.draw.circle(r=p['cx'], c=p['cy'], radius=p['r'])
            elif p["name"] == "rect":
                start = (p['x'], p['y'])
                extent = (p['width'], p['height'])
                rr, cc = skimage.draw.rectangle(start=start, extent=extent)
            elif p["name"] == "ellipse":
                rr, cc = skimage.draw.ellipse(r=p['cx'], c=p['cy'], r_radius=p['rx'], c_radius=p[
                    'ry'], rotation=np.deg2rad(p['theta']))
            else:
                rr, cc = skimage.draw.polygon(r=p['all_points_y'], c=p['all_points_x'])

            if len(region_info['Elevator Item']) == 1:
                # mask for button or label
                class_id = 1 if region_info['Elevator Item']['Button'] else 2
            else:
                # mask for a button and label
                class_id = 3

            m[rr, cc, i] = class_id
            instance_masks.append(np.ma.make_mask(m))
            class_ids.append(class_id)

        masks = np.stack(instance_masks, axis=2).astype(np.bool)
        class_ids = np.array(class_ids, dtype=np.int32)

        return masks, class_ids

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "elevator_panel":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
