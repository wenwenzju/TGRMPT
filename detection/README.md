# TGRMPT: Detection
Currently, we use [yolov5](https://github.com/ultralytics/yolov5) as our pedestrian detection method.
Change working directory to TGRMPT/detection/yolov5.
## Preparing Detection Dataset
First, download our TGRMPT dataset. See [root README](../README.md).

Second, Generate detection images according to coco dataset format from our MOT dataset, running the following command:
```shell
python convert_mot_to_coco_detection.py --data-path ../../dataset
```
The above command will generate detection dataset, both for whole body and head shoulder, in `../../dataset`. Note, the images in `detection` are soft links to images in `mot`.
The detection directory structure is like:
```
├── head_shoulder
│   ├── images -> /home/wenwenzju/Tmp/iros2022/detection/images
│   └── labels
│       ├── test
│       └── train
├── images
│   ├── test (185813 images)
│   └── train (228366 images)
└── whole_body
    ├── images -> /home/wenwenzju/Tmp/iros2022/detection/images
    └── labels
        ├── test
        └── train
```

## Train
To train whole body detector, run
```shell
bash train_wb.sh
```
The above command will save logs to `runs/wb`.

To train head shoulder detector, run
```shell
bash train_hs.sh
```
The above command will save logs to `runs/hs`.

## Create Detections
To generate detection results of test images, run
```shell
bash create_wb_detection.sh    # whole body detection results
bash create_hs_detection.sh    # head shoulder detection results
```
The above command will generate detection results in `dataset/mot/mot17/${sequence}/detection`. The results are used to extract embedding features in the subsequent step.

## Deploy
To deploy the trained whole body and head shoulder models to onnx, run
```shell
bash deploy_onnx.sh
```
To deploy the trained whole body and head shoulder models to TensorRT, run
```shell
bash deploy_trt.sh
```
The converted models will be saved in `runs/hs/exp/weights` for head shoulder and `runs/wb/exp/weights` for whole body.