import json


annotations = json.load(open("elevator_panels/train/masks.json"))

for a in annotations:
    print(annotations[a]["filename"])
    polygons = [r['shape_attributes'] for r in annotations[a]['regions']]
    ids = []
    for region in annotations[a]["regions"]:
        # print(region)
        ra = region["region_attributes"]
        if ra["Elevator Item"] == "Button":
            class_id = 2
        else:
            class_id = 1
        ids.append(class_id)

    print(ids)
