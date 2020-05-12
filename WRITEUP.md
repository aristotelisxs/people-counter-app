# People counter application

Using OpenCV and Intel's OpenVino Toolkit, this project is able to count people and deliver statistics on their time on 
camera through a lightweight YOLOv3-tiny model at the edge. In this write-up, we discuss on the model's performance and 
its suitability in running on low resource devices.

## Explaining Custom Layers

Converting models through the OpenVino toolkit to their Intermediate Representation (IR) requires that all neural 
network layers are supported by the toolkit. If there are layers that are not, these are considered as "custom" from the
 perspective of the toolkit. To convert these layers we need to register them first as extensions in order for them to 
 be optimized into the IR. The steps of doing so are as follows:

* Generate templates for the Model Optimizer and Inference Engine (CPU and/or GPU)
* List all parameters needed as input to the custom layers
* Modify the extractor and operation extension source files to list the functionality of the custom layers and,
* Generate the IR files from the above

Custom layers serve multiple purposes. They could represent new activation functions that work best with the problem at 
hand. They could also be new types of layers developed during research endeavours for AI at the edge. The possibilities 
are endless with these, as the field of Deep Learning is ever-growing. Handling custom layers is a way for the toolkit 
to be able to catch up with latest trends.

## Setup

### DL Workbench

I have used a docker container to setup and access the DL Workbench. The full instructions, based on your device, can be
found [here](https://docs.openvinotoolkit.org/2020.1/_docs_Workbench_DG_Install_from_Docker_Hub.html#install_dl_workbench_from_docker_hub_on_windows_os).
Since I have a Windows device, I had to execute the following after pulling the docker image:

`winpty docker run -p 127.0.0.1:5665:5665 --name workbench -e PROXY_HOST_ADDRESS=0.0.0.0 -e PORT=5665 -it openvino/workbench:latest`

### YOLOv3-tiny model

The `.pb` format of the model was downloaded from [this](https://github.com/mystic123/tensorflow-yolo-v3) Github repository.
Download the weights and class names using:

```buildoutcfg
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
wget https://pjreddie.com/media/files/yolov3-tiny.weights
```

Then, you should be able to generate the frozen graph model by:
`python convert_weights_pb.py --class_names coco.names --data_format NHWC --weights_file yolov3-tiny.weights --tiny`

To convert the resulting `.pb` model file we have run the following (on the Udacity workspace):
```buildoutcfg
export MO_ROOT=/opt/intel/openvino/deployment_tools/model_optimizer/
python3 $MO_ROOT/mo_tf.py --input_model frozen_darknet_yolov3_model_tiny.pb --tensorflow_use_custom_operations_config $MO_ROOT/extensions/front/tf/yolo_v3_tiny.json --batch 1
```

Make sure that the environment variable points to the correct OpenVino installation folder and that the `.pb` has the 
name: `frozen_darknet_yolov3_model_tiny.pb`

## Comparing Model Performance

My method to compare models post-conversion involved re-purposing the CUHK person re-identification dataset, available 
[here](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html) (under the name CUHK01). Download and unzip the 
images of containing students walking across campus (one person per image). We have selected this dataset as it resembles
the exercise's objective of people counting (one at a time). 

To assess post-model conversion performance, we have used the `benchmark.py` script with the following arguments:
`python benchmark.py -m path/to/frozen/xml/file -img_dir path/to/cuhk01/campus/directory -pt 0.01 
-l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so`

To assess pre-model conversion performance, we have forked the repository where the downloaded model was and modified it
to accept multiple images from a directory. Find this new repository 
[here](https://github.com/aristotelisxs/tensorflow-yolo-v3). To produce the results, execute the following:

`python demo.py --max_imgs=100 --input_dir path/to/cuhk01/campus/directory --conf_threshold 0.01 
--frozen_model path/to/frozen/pb/file`

Notice that the probability threshold is set very low when running the benchmark scripts to allow collecting for scores
from all ranges.

The benchmark results, run across CUHK 100 images, can be seen below:

Model Type                | Model Size | Average Inference Time | Average score (out of 100) | Maximum score (out of 100)
--------------------------|------------|-------------------------|----------------------|------------
frozen_darknet_yolov3_model_tiny.pb | 33.8 MB     | 1.32 seconds             | 55.72          | 93.94
FP32 OpenVINO IR          | 33.7 MB     | 89.72 milliseconds             | 8.77          | 58.9
 
The scores here disregard instances where a person fails to be recorded by the algorithms (despite all images including 
a single person each). The gap between the two is large with this benchmark and the accuracy of the pre-conversion model
is not great either. We believe this to be because of a generalization issue of the model in other datasets than the one
it was trained on (COCO) (in terms of image sizes, image quality etc.).

## Assess Model Use Cases

Some of the potential use cases of the people counter app include, but are not limited to:
* Burglar house intrusion alerting
* Counting people within a retail store and alerting whether tha maximum capacity has been reached or is allowed (due to
 COVID-19 government guidelines)
* Ensuring that kids are monitored in their playground and, if any are missing, sound an alert.
* Buildings with fire codes allow only a certain number of people in large meeting rooms. To keep track of that, an 
automated people counter could be enforced.
* A people counter app could help museums, restaurants and/or stores know if there are people that still have not exited
the building and lock up safely for the night.

Each of these use cases covers security measures that, when breached, the authorities or the people responsible for a 
specific situation are alerted in time. In this way, we can aid first-line workers at critical situations in a rapid manner.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. 
The potential effects of each of these are as follows:

Image processing software is sensitive to contrast in the image. To achieve adequate contrast, an artificial light source could 
help illuminate objects in the scene. Assessing the efficiency of this under different lighting conditions would help developers
understand whether the conditions are acceptable or not, disregarding model predictions entirely where not.

False positives/negatives in low accuracy model scenarios can greatly hinder the ability of our people counter app. 
Having a margin of error into consideration can help system designs to tolerate these wrong outputs at a rate where the
application is not rendered completely useless.  

High-quality images are fundamental for image processing. The difference in image quality directly affects accuracy when
 using image processing technology. Camera selection according to the application is also 
important. If possible, the training set should be based on the same types of cameras that the model is deployed to. 

Scenes captured with short-focal-length lenses appear expanded in depth, while those captured with long lenses appear 
compressed/blurry. Higher quality image give the model more information but require more neural network nodes and more 
computing power to process.

In general, a model used might be trained for specific input image and target size. So, having a drastically different 
input image size and scaling it to the required size might have a negative impact on the performance.

## References

[1] OpenVino Toolkit. 2020. Deep Learning Workbench Developer Guide. [ONLINE] Available at: https://docs.openvinotoolkit.org/2020.1/_docs_Workbench_DG_Introduction.html. [Accessed 8 May 2020].
[2] Redmon, Joseph, and Ali Farhadi. "Yolov3: An incremental improvement." arXiv preprint arXiv:1804.02767 (2018).
[3] Joseph Chet Redmon. 2020. YOLO: Real-Time Object Detection. [ONLINE] Available at: https://pjreddie.com/darknet/yolo/. [Accessed 8 May 2020].
