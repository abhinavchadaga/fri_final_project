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
        mask = np.zeros([info["height"], info["width"], len(info["masks"])],
                        dtype=np.uint8)
