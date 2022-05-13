import cv_pipeline

correct = 0
button_ops = ["Mezzanine", "Floor 2", "Floor 1",
              "Door Hold", "Open Door", "Close Door", "Phone", "Bell"]

cv_pipeline.detect(
    "./elevator_panels/val/{file}.jpg".format(file=4), "Mezzanine")
