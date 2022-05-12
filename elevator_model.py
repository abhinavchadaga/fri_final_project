from mrcnn.config import Config
import mrcnn.model as modellib

MODEL_DIR = "./logs"


class ElevatorPanelConfig(Config):
    NAME = "elevator_panel"

    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 2

    STEPS_PER_EPOCH = 100

    DETECTION_MIN_CONFIDENCE = 0.9

    LEARNING_RATE = 0.0007


config = ElevatorPanelConfig()


class InferenceConfig(ElevatorPanelConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


inference_config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

path_to_weights = "./mask_rcnn_elevator_panel.h5"


# Load trained weights
print("Loading weights from ", path_to_weights)
model.load_weights(path_to_weights, by_name=True)
