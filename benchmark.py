"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import time
import cv2
import numpy as np

from inference import Network
from argparse import ArgumentParser
from yolo_v3_utils import ParseYOLOV3Output, IntersectionOverUnion


LABELS = ("person", "bicycle", "car", "motorbike", "aeroplane",
          "bus", "train", "truck", "boat", "traffic light",
          "fire hydrant", "stop sign", "parking meter", "bench", "bird",
          "cat", "dog", "horse", "sheep", "cow",
          "elephant", "bear", "zebra", "giraffe", "backpack",
          "umbrella", "handbag", "tie", "suitcase", "frisbee",
          "skis", "snowboard", "sports ball", "kite", "baseball bat",
          "baseball glove", "skateboard", "surfboard","tennis racket", "bottle",
          "wine glass", "cup", "fork", "knife", "spoon",
          "bowl", "banana", "apple", "sandwich", "orange",
          "broccoli", "carrot", "hot dog", "pizza", "donut",
          "cake", "chair", "sofa", "pottedplant", "bed",
          "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
          "remote", "keyboard", "cell phone", "microwave", "oven",
          "toaster", "sink", "refrigerator", "book", "clock",
          "vase", "scissors", "teddy bear", "hair drier", "toothbrush")


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=False, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                             "(0.5 by default)")
    parser.add_argument("-iou", "--iou_threshold",
                        help="Intersection over union threshold"
                             "(0.4 by default)", type=float, default=0.2)
    parser.add_argument("-img_dir", "--image_directory",
                        help="The path to the directory of images. Ignores --input if set", type=str,
                        required=False)
    parser.add_argument("-max_imgs", "--maximum_images",
                        help="Stop at this much images processed", type=int, required=False, default=100)

    return parser


def pre_process(frame, new_w, new_h, net_h, net_w):
    p_frame = cv2.resize(frame, (new_w, new_h))
    canvas = np.full((net_h, net_w, 3), 128)
    canvas[(net_h - new_h) // 2:
           (net_h - new_h) // 2 + new_h, (net_w - new_w) // 2:
                                         (net_w - new_w) // 2 + new_w, :] = p_frame
    pp_img = canvas
    pp_img = pp_img.transpose((2, 0, 1))
    pp_img = pp_img.reshape(1, *pp_img.shape)  # Batch size axis add & NHWC to NCHW

    return pp_img


def post_conversion_benchmark(frame, model, cpu_extension, device, prob_threshold, iou_threshold, network,
                              net_input_shape):
    height, width, channels = frame.shape

    net_w = net_input_shape[2]
    net_h = net_input_shape[3]

    new_w = int(width * min(net_w / width, net_h / height))
    new_h = int(height * min(net_w / width, net_h / height))

    pp_img = pre_process(frame, new_w, new_h, net_h, net_w)

    inference_start_time = time.time()
    network.exec_net(pp_img)
    if network.wait() == 0:
        inference_end_time = time.time()
        total_inference_time = inference_end_time - inference_start_time

        outputs = network.get_output()
        objects = []
        for output in outputs.values():
            objects = ParseYOLOV3Output(output, new_h, new_w, height, width, prob_threshold, objects)

        # Filtering overlapping boxes
        obj_len = len(objects)
        for i in range(obj_len):
            if objects[i].confidence == 0.0:
                continue
            for j in range(i + 1, obj_len):
                if IntersectionOverUnion(objects[i], objects[j]) >= iou_threshold:
                    objects[j].confidence = 0

        # Drawing boxes
        conf = 0
        for obj in objects:
            if obj.confidence < prob_threshold:
                continue
            lbl = obj.class_id
            label = LABELS[lbl]
            if label != 'person':
                continue
            else:
                conf = obj.confidence * 100
                break

        return conf, round(total_inference_time * 1000, 3)


def main():
    args = build_argparser().parse_args()

    scores = list()
    scores_wout_mispredictions = list()
    inference_time = list()

    network = Network()
    network.load_model(args.model, cpu_extension=args.cpu_extension, device=args.device)
    net_input_shape = network.get_input_shape()

    if args.image_directory is not None:
        if os.path.exists(args.image_directory):
            for root, dirs, files in os.walk(args.image_directory, topdown=False):
                counter = 0
                for name in files:
                    if counter >= args.maximum_images:
                        break

                    frame = cv2.imread(os.path.join(root, name))
                    res, inf_time = post_conversion_benchmark(frame, args.model, args.cpu_extension, args.device,
                                                              args.prob_threshold, args.iou_threshold, network,
                                                              net_input_shape)

                    if res > 0:
                        scores_wout_mispredictions.append(res)

                    scores.append(res)
                    inference_time.append(inf_time)

                    counter += 1
    else:
        res, inf_time = post_conversion_benchmark(args.input_img, args.model, args.cpu_extension, args.device,
                                                              args.prob_threshold, args.iou_threshold, network,
                                                              net_input_shape)

        if res > 0:
            scores_wout_mispredictions.append(res)

        scores.append(res)
        inference_time.append(inf_time)

    print("Average score across all images: " + str(np.mean(scores)))
    print("Max score across all images: " + str(np.max(scores)))
    print("Average inference time: " + str(np.mean(inference_time)) + "ms")
    print("Average score disregarding mis-predictions: " + str(np.mean(scores_wout_mispredictions)))
    print("Minimum score disregarding mis-predictions: " + str(np.min(scores_wout_mispredictions)))


if __name__ == '__main__':
    main()
