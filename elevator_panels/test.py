import json

# str = """{"120.jpg100167": {
#         "filename": "120.jpg",
#         "size": 100167,
#         "regions": [
#             { "shape_attributes": { "name": "circle", "cx": 183, "cy": 125, "r": 38 }, "region_attributes": { "Elevator Item": { "Button": true } } },
#             { "shape_attributes": { "name": "circle", "cx": 361, "cy": 125, "r": 39 }, "region_attributes": { "Elevator Item": { "Button": true } } },
#             { "shape_attributes": { "name": "circle", "cx": 363, "cy": 227, "r": 38.552 }, "region_attributes": { "Elevator Item": { "Button": true } } },
#             { "shape_attributes": { "name": "circle", "cx": 361, "cy": 327, "r": 36 }, "region_attributes": { "Elevator Item": { "Button": true } } },
#             { "shape_attributes": { "name": "circle", "cx": 363, "cy": 427, "r": 36 }, "region_attributes": { "Elevator Item": { "Button": true } } },
#             { "shape_attributes": { "name": "circle", "cx": 185, "cy": 428, "r": 35.014 }, "region_attributes": { "Elevator Item": { "Button": true } } },
#             { "shape_attributes": { "name": "circle", "cx": 186, "cy": 327, "r": 37 }, "region_attributes": { "Elevator Item": { "Button": true } } },
#             { "shape_attributes": { "name": "circle", "cx": 184, "cy": 228, "r": 36 }, "region_attributes": { "Elevator Item": { "Button": true } } },
#             { "shape_attributes": { "name": "circle", "cx": 181, "cy": 21, "r": 39.003 }, "region_attributes": { "Elevator Item": { "Button": true } } },
#             { "shape_attributes": { "name": "circle", "cx": 362, "cy": 20, "r": 37 }, "region_attributes": { "Elevator Item": { "Button": true } } },
#             {
#                 "shape_attributes": { "name": "polyline", "all_points_x": [145, 145, 110, 96, 82, 76, 76, 81, 89, 101, 115, 129, 144], "all_points_y": [186, 269, 269, 263, 250, 233, 220, 207, 196, 187, 185, 185, 185] },
#                 "region_attributes": { "Elevator Item": { "Label": true } }
#             },
#             {
#                 "shape_attributes": { "name": "polyline", "all_points_x": [142, 141, 109, 96, 87, 74, 73, 76, 85, 100, 114, 132, 138], "all_points_y": [82, 168, 168, 164, 155, 140, 119, 106, 93, 84, 82, 82, 82] },
#                 "region_attributes": { "Elevator Item": { "Label": true } }
#             },
#             {
#                 "shape_attributes": { "name": "polyline", "all_points_x": [111, 140, 140, 72, 69, 68, 70, 76, 85, 96, 107], "all_points_y": [64, 64, 1, 0, 10, 22, 33, 43, 53, 60, 63] },
#                 "region_attributes": { "Elevator Item": { "Label": true } }
#             },
#             {
#                 "shape_attributes": { "name": "polyline", "all_points_x": [143, 144, 110, 96, 82, 76, 78, 82, 94, 109, 129, 140], "all_points_y": [286, 370, 370, 364, 350, 332, 316, 305, 293, 288, 286, 286] },
#                 "region_attributes": { "Elevator Item": { "Label": true } }
#             },
#             {
#                 "shape_attributes": { "name": "polyline", "all_points_x": [144, 145, 108, 95, 84, 78, 79, 85, 96, 110, 122, 139], "all_points_y": [386, 469, 469, 462, 450, 436, 418, 404, 394, 388, 387, 387] },
#                 "region_attributes": { "Elevator Item": { "Label": true } }
#             },
#             {
#                 "shape_attributes": { "name": "polyline", "all_points_x": [318, 318, 286, 272, 262, 256, 254, 258, 268, 286, 307], "all_points_y": [81, 164, 165, 159, 149, 138, 119, 105, 92, 82, 81] },
#                 "region_attributes": { "Elevator Item": { "Label": true } }
#             },
#             {
#                 "shape_attributes": { "name": "polyline", "all_points_x": [322, 288, 273, 262, 256, 253, 252, 253, 256, 321, 322], "all_points_y": [62, 62, 56, 48, 38, 27, 18, 6, 2, 2, 60] },
#                 "region_attributes": { "Elevator Item": { "Label": true } }
#             },
#             {
#                 "shape_attributes": { "name": "polyline", "all_points_x": [323, 323, 286, 271, 260, 254, 254, 258, 265, 275, 286, 296, 312, 319], "all_points_y": [185, 265, 266, 260, 249, 235, 220, 205, 195, 188, 185, 185, 185, 185] },
#                 "region_attributes": { "Elevator Item": { "Label": true } }
#             },
#             {
#                 "shape_attributes": { "name": "polyline", "all_points_x": [319, 320, 286, 274, 264, 257, 255, 254, 257, 264, 277, 286, 304, 312], "all_points_y": [286, 366, 366, 362, 354, 343, 331, 321, 310, 299, 290, 286, 286, 286] },
#                 "region_attributes": { "Elevator Item": { "Label": true } }
#             },
#             {
#                 "shape_attributes": { "name": "polyline", "all_points_x": [320, 322, 286, 274, 260, 255, 255, 260, 269, 278, 292, 318], "all_points_y": [384, 466, 466, 461, 449, 434, 418, 404, 394, 388, 384, 383] },
#                 "region_attributes": { "Elevator Item": { "Label": true } }
#             },
#             {
#                 "shape_attributes": { "name": "polyline", "all_points_x": [491, 461, 440, 435, 432, 435, 441, 453, 463, 475, 500, 501], "all_points_y": [184, 184, 197, 208, 224, 241, 251, 261, 265, 267, 267, 186] },
#                 "region_attributes": { "Elevator Item": { "Label": true } }
#             },
#             {
#                 "shape_attributes": { "name": "polyline", "all_points_x": [498, 463, 450, 438, 433, 431, 433, 437, 445, 455, 468, 501, 501], "all_points_y": [81, 81, 88, 100, 111, 123, 133, 145, 152, 160, 164, 163, 83] },
#                 "region_attributes": { "Elevator Item": { "Label": true } }
#             },
#             {
#                 "shape_attributes": {
#                     "name": "polyline",
#                     "all_points_x": [501, 463, 450, 441, 435, 431, 431, 433, 438, 445, 455, 467, 486, 502, 501],
#                     "all_points_y": [283, 284, 287, 295, 306, 319, 330, 343, 351, 358, 364, 368, 368, 367, 285]
#                 },
#                 "region_attributes": { "Elevator Item": { "Label": true } }
#             },
#             {
#                 "shape_attributes": { "name": "polyline", "all_points_x": [492, 465, 451, 439, 433, 430, 429, 432, 439, 449, 460, 481, 497, 501], "all_points_y": [383, 384, 388, 397, 409, 419, 429, 440, 452, 461, 467, 467, 467, 387] },
#                 "region_attributes": { "Elevator Item": { "Label": true } }
#             }
#         ],
#         "file_attributes": {}
#     }}"""

f = open("elevator_panels/train/masks.json")
annotations = json.loads(f.read())


for a in annotations:
    masks = [region["shape_attributes"]
             for region in annotations[a]["regions"]]

    class_ids = []
    for region in annotations[a]["regions"]:
        ei = region["region_attributes"]["Elevator Item"]
        class_id = 1 if 'Label' in ei else 2
        class_ids.append(class_id)

    print(masks)
    print(class_ids)
