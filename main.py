import cv_pipeline

correct = 0
button_ops = ["Mezzanine", "Floor 2", "Floor 1",
              "Door Hold", "Open Door", "Close Door", "Phone", "Bell"]
# count = 0
# correct = 0
# for i in range(11):
#     for op in button_ops:
#         count += 1
#         correct += cv_pipeline.detect(
#             "./elevator_panels/val/{file}.jpg".format(file=i), op)
#         print("correct: ", correct, " total: ", count)

cv_pipeline.detect(
    "./elevator_panels/val/{file}.jpg".format(file=0), "Floor 1")

# print("result: ", correct / count)
