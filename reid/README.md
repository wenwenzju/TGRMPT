# Embedding Feature Extracting
Currently, we use [fast-reid](https://github.com/JDAI-CV/fast-reid) platform to train our ReID models. Change working directory to TGRMPT/reid/fast-reid/projects/iros2022
## Preparing ReID Dataset
Download our TGRMPT dataset. See [root README](../README.md).

## Train
To train whole body ReID model, run
```shell
bash train_whole_body.sh 0
```
The above command will save logs and model parameters to `logs/bot_r18_train_on_iros2022_fisheye_whole_body`.

To train head shoulder ReID model, run
```shell
bash train_head_shoulder.sh 0
```
The above command will save logs and model parameters to `logs/bot_r18_train_on_iros2022_fisheye_head_shoulder`.

## Extract Features
To extract embedding features of all detections in test images, run
```shell
bash extract_wb_features.sh 0 4   # extract whole body features. The first param is gpu id, and the second is number of processes.
bash extract_hs_features.sh 0 4   # extract head shoulder features. The first param is gpu id, and the second is number of processes.
```
The above command will save detected bounding boxes and the corresponding embedding features in `dataset/mot/mot17/${sequence}/embedding`. The results are used to perform tracking.

## Deploy
To deploy the trained whole body and head shoulder ReID models to onnx, run
```shell
bash deploy_onnx.sh
```
To deploy the trained whole body and head shoulder models to TensorRT, run
```shell
bash deploy_trt.sh
```
The converted models will be saved in `logs/bot_r18_train_on_iros2022_fisheye_head_shoulder` for head shoulder and `logs/bot_r18_train_on_iros2022_fisheye_whole_body` for whole body.
