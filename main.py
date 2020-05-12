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


import sys
import time
import socket
import json
import cv2

import numpy as np
import paho.mqtt.client as mqtt


from sys import platform
from inference import Network
from collections import deque
from argparse import ArgumentParser
from datetime import datetime, timedelta
from yolo_v3_utils import ParseYOLOV3Output, IntersectionOverUnion

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

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

label_text_color = (255, 255, 255)
label_background_color = (125, 175, 75)
box_color = (255, 128, 0)
box_thickness = 1

if platform == "linux" or platform == "linux2":
    CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
    CODEC = 0x00000021
elif platform == "darwin":
    CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension.dylib"
    CODEC = cv2.VideoWriter_fourcc('M','J','P','G')
else:
    print("Unsupported OS.")
    exit(1)


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=CPU_EXTENSION,
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
    parser.add_argument("-dt", "--detect_threshold", default=2, type=int,
                        help="How many seconds should we wait after the model detects nothing within the frame to "
                             "communicate this? This is to tolerate false positives during inference.")
    parser.add_argument("-iou", "--intersection_over_union_threshold", type=float, default=0.2,
                        help="The percentage of overlap above which parts of an image should merge "
                             "(applicable for YOLO models)")

    return parser


def connect_mqtt():
    """
    Connect to the MQTT client to post people counting statistics
    """
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()

    infer_network.load_model(model=args.model, device="CPU", cpu_extension=args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()
    net_w = net_input_shape[2]
    net_h = net_input_shape[3]
    
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    # yolo v3 inputs
    new_w = int(width * min(net_w/width, net_h/height))
    new_h = int(height * min(net_w/width, net_h/height))
    
    # Create a video writer for the output video
    # out = cv2.VideoWriter('out_' + str(cur_ts) +'.mp4', CODEC, 30, (width, height))
    out = cv2.VideoWriter('out.mp4', CODEC, 30, (width, height))

    tolerance_start_time = datetime.now()

    total_people_counted = 0
    tolerance_time = timedelta(seconds=args.detect_threshold)
    prev_total_people = 0
    prev_duration = 0
    cur_time = datetime.now()
    fps_dq = deque(maxlen=100)

    while cap.isOpened():
        flag, frame = cap.read()
        t1 = time.time()

        if not flag:
            break

        key_pressed = cv2.waitKey(60)

        if key_pressed == 27:
            break
        
        p_frame = cv2.resize(frame, (new_w, new_h))
        canvas = np.full((net_h, net_w, 3), 128)
        canvas[(net_h-new_h) // 2:
               (net_h-new_h) // 2 + new_h, (net_w-new_w) // 2:
                                           (net_w-new_w)//2 + new_w, :] = p_frame
        pp_img = canvas
        pp_img = pp_img.transpose((2, 0, 1))
        pp_img = pp_img.reshape(1, *pp_img.shape)  # Batch size axis add & NHWC to NCHW

        # A-synchronous (non-blocking) inference
        infer_network.exec_net(pp_img)

        if infer_network.wait() == 0:
            outputs = infer_network.get_output()

            objects = []
            for output in outputs.values():
                objects = ParseYOLOV3Output(output, new_h, new_w, height, width, args.prob_threshold, objects)

            # Filtering overlapping boxes
            obj_len = len(objects)
            for i in range(obj_len):
                if objects[i].confidence == 0.0:
                    continue
                for j in range(i + 1, obj_len):
                    if IntersectionOverUnion(objects[i], objects[j]) >= args.intersection_over_union_threshold:
                        objects[j].confidence = 0

            # Drawing boxes
            cur_total_people = 0
            for obj in objects:
                if obj.confidence < args.prob_threshold:
                    continue
                label = obj.class_id
                confidence = obj.confidence
                lbl = LABELS[label]
                if confidence > args.prob_threshold and lbl == "person":
                    label_text = LABELS[label] + " (" + "{:.1f}".format(confidence * 100) + "%)"
                    cv2.rectangle(frame, (obj.xmin, obj.ymin), (obj.xmax, obj.ymax), box_color,
                                  box_thickness)
                    cv2.putText(frame, label_text, (obj.xmin, obj.ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                label_text_color, 1)
                    cur_total_people += 1

            if cur_total_people > prev_total_people and tolerance_start_time + tolerance_time <= datetime.now():
                prev_total_people = cur_total_people
                tolerance_start_time = datetime.now()
                cur_time = datetime.now()

            if cur_total_people == prev_total_people:
                # Update current time
                tolerance_start_time = datetime.now()

            if cur_total_people < prev_total_people and tolerance_start_time + tolerance_time <= datetime.now():
                duration = int((datetime.now() - cur_time).seconds)

                diff = prev_total_people - cur_total_people
                total_people_counted += diff

                cur_duration = int(duration / diff)
                cur_duration = int(cur_duration if prev_duration == 0 else cur_duration + prev_duration / 2)

                client.publish("person/duration", json.dumps({"duration": cur_duration}))
                tolerance_start_time = datetime.now()
                prev_total_people = cur_total_people
                prev_duration = cur_duration
                cur_time = datetime.now()

            client.publish("person", json.dumps({"count": cur_total_people}))

            elapsed_time = time.time() - t1
            fps = "(Playback) {:.1f} FPS".format(1 / elapsed_time)
            fps_dq.append(1 / elapsed_time)
            cv2.putText(frame, fps, (width - 170, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38, 0, 255), 1, cv2.LINE_AA)

            # Write out the frame
            out.write(frame)

        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        if len(fps_dq) == 100:
            with open('stats.txt', 'w+') as f:
                f.write('Average FPS over last ' + str(len(fps_dq)) + ' frames: ' + str(np.mean(fps_dq)))

            f.close()

            fps_dq = deque(maxlen=100)

        if not args.input.endswith(".mp4"):  # Assume only mp4 files will be given as video inputs
            cv2.imwrite('test.png', frame)

    # Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()


def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
