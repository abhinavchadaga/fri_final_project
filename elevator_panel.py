# from pickletools import uint8
# from mrcnn import model as modellib, utils
# from mrcnn.config import Config
# import json
# import os
# from re import S
# import sys
# import time
# import numpy as np
# import imgaug
# import skimage.io
from mrcnn import model as modellib, utils
from mrcnn.config import Config
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

ROOT_DIR = os.path.abspath("./")


sys.path.append(ROOT_DIR)

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

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes']
                            for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]

            ids = []
            for region in a["regions"]:
                # if len(region["region_attributes"]["Elevator Item"]) == 2:
                #     class_id = 3
                # else:
                ei = region["region_attributes"]["Elevator Item"]
                if "Button" in ei and ei["Button"] == True and "Label" in ei and ei["Label"] == True:
                    class_id = 3
                elif "Button" in ei and ei["Button"]:
                    class_id = 2
                else:
                    class_id = 1
                # class_id = 1 if 'Label' in ei else 2
                ids.append(class_id)

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "elevator_panel",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                ids=ids)

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        polygons = image_info["polygons"]
        count = len(polygons)
        if image_info["source"] != "elevator_panel":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]

        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], count],
                        dtype=np.uint8)

        print('img height: ', info['height'])
        print('img width: ', info['width'])

        for i, p in enumerate(polygons):
            # Get indexes of pixels inside the polygon and set them to 1
            # rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            if p["name"] == "circle":
                rr, cc = skimage.draw.circle(
                    r=p['cy'], c=p['cx'], radius=p['r'])
            elif p["name"] == "rect":
                start = (p['y'], p['x'])
                extent = (p['height'], p['width'])
                rr, cc = skimage.draw.rectangle(start=start, extent=extent)
            elif p["name"] == "ellipse":
                rr, cc = skimage.draw.ellipse(r=p['cy'], c=p['cx'], r_radius=p['ry'], c_radius=p[
                    'rx'], rotation=np.deg2rad(p['theta']))
            else:
                rr, cc = skimage.draw.polygon(
                    r=p['all_points_y'], c=p['all_points_x'])

            mask[rr, cc, i] = 1

        class_ids = np.array(image_info["ids"], dtype=np.int32)
        print(class_ids)

        # np.ones([mask.shape[-1]], dtype=np.int32)
        return mask.astype(np.bool), class_ids

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "elevator_panel":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = ElevatorPanelDataset()
    dataset_train.load_elevator(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ElevatorPanelDataset()
    dataset_val.load_elevator(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect elevator panel buttons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Elevator Panel dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = ElevatorPanelConfig()
    else:
        class InferenceConfig(ElevatorPanelConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = PATH_TO_COCO_WEIGHTS
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    # elif args.command == "splash":
    #     detect_and_color_splash(model, image_path=args.image,
    #                             video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
