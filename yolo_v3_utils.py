"""
Disclaimer. The below are directly referenced from https://github.com/jjeamin/OpenVINO-YoloV3/blob/master/openvino_yolov3_test.py
All credits go to its respective author.
"""

import numpy as np, math

yolo_scale_13 = 13
yolo_scale_26 = 26
yolo_scale_52 = 52

N_CLASSES = 80
COORDS = 4
num = 3
ANCHORS = [10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319]


def EntryIndex(side, lcoords, lclasses, location, entry):
    n = int(location / (side * side))
    loc = location % (side * side)
    return int(n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc)


class DetectionObject:
    xmin = 0
    ymin = 0
    xmax = 0
    ymax = 0
    class_id = 0
    confidence = 0.0

    def __init__(self, x, y, h, w, class_id, confidence, h_scale, w_scale):        
        self.xmin = int((x - w / 2) * w_scale)
        self.ymin = int((y - h / 2) * h_scale)
        self.xmax = int(self.xmin + w * w_scale)
        self.ymax = int(self.ymin + h * h_scale)
        self.class_id = class_id
        self.confidence = confidence


def IntersectionOverUnion(box_1, box_2):
    width_of_overlap_area = min(box_1.xmax, box_2.xmax) - max(box_1.xmin, box_2.xmin)
    height_of_overlap_area = min(box_1.ymax, box_2.ymax) - max(box_1.ymin, box_2.ymin)

    if width_of_overlap_area < 0.0 or height_of_overlap_area < 0.0:
        area_of_overlap = 0.0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area

    box_1_area = (box_1.ymax - box_1.ymin) * (box_1.xmax - box_1.xmin)
    box_2_area = (box_2.ymax - box_2.ymin) * (box_2.xmax - box_2.xmin)
    area_of_union = box_1_area + box_2_area - area_of_overlap

    if area_of_union <= 0.0:
        ret_val = 0.0
    else:
        ret_val = (area_of_overlap / area_of_union)

    return ret_val


def ParseYOLOV3Output(blob, resized_im_h, resized_im_w, original_im_h, original_im_w, threshold, objects):

    out_blob_h = blob.shape[2]

    side = out_blob_h
    anchor_offset = 0

    if len(ANCHORS) == 18:   ## YoloV3
        if side == yolo_scale_13:
            anchor_offset = 2 * 6
        elif side == yolo_scale_26:
            anchor_offset = 2 * 3
        elif side == yolo_scale_52:
            anchor_offset = 2 * 0

    elif len(ANCHORS) == 12: ## tiny-YoloV3
        if side == yolo_scale_13:
            anchor_offset = 2 * 3
        elif side == yolo_scale_26:
            anchor_offset = 2 * 0

    else:                    ## ???
        if side == yolo_scale_13:
            anchor_offset = 2 * 6
        elif side == yolo_scale_26:
            anchor_offset = 2 * 3
        elif side == yolo_scale_52:
            anchor_offset = 2 * 0

    side_square = side * side
    output_blob = blob.flatten()

    for i in range(side_square):
        row = int(i / side)
        col = int(i % side)
        for n in range(num):
            obj_index = EntryIndex(side, COORDS, N_CLASSES, n * side * side + i, COORDS)
            box_index = EntryIndex(side, COORDS, N_CLASSES, n * side * side + i, 0)
            scale = output_blob[obj_index]
            if scale < threshold:
                continue
            x = (col + output_blob[box_index + 0 * side_square]) / side * resized_im_w
            y = (row + output_blob[box_index + 1 * side_square]) / side * resized_im_h
            h_exp = math.exp(output_blob[box_index + 3 * side_square]) # if math.exp(output_blob[box_index + 3 * side_square]) is np.nan else 1
            height = h_exp * ANCHORS[anchor_offset + 2 * n + 1]
            w_exp = math.exp(output_blob[box_index + 2 * side_square])  # if math.exp(output_blob[box_index + 2 * side_square]) is np.nan else 1
            width = w_exp * ANCHORS[anchor_offset + 2 * n]
            for j in range(N_CLASSES):
                class_index = EntryIndex(side, COORDS, N_CLASSES, n * side_square + i, COORDS + 1 + j)
                prob = scale * output_blob[class_index]
                if prob < threshold:
                    continue
                obj = DetectionObject(x, y, height, width, j, prob, (original_im_h / resized_im_h), (original_im_w / resized_im_w))
                objects.append(obj)

    return objects
