# Tracking
Currently, we use [deep_sort](https://github.com/nwojke/deep_sort) as our tracking framework. To detect pedestrians, we use yolov5s trained on our dataset, see [README.md](../detection/README.md).
To extract embedding features, we use ResNet18 ReID model trained on our dataset, see [README.md](../reid/README.md).
Change working directory to `TGRMPT/tracking`.
## Preparing ReID Dataset
First, download our TGRMPT dataset. See [root README](../README.md).

Second, run the following shell commands to generate MOT ground truth folder prepared for evaluation. We use [TrackEval](https://github.com/JonathonLuiten/TrackEval) as our evaluation toolkit.
```shell
cd eval
mkdir -p data/gt/zjlab/iros2022-fisheye-similar-test
mkdir data/gt/zjlab/iros2022-fisheye-tradition-test
mkdir data/gt/zjlab/seqmaps
cd data/gt/zjlab/iros2022-fisheye-similar-test
ls -d ../../../../../../dataset/mot/mot17/{02,06,10,12,14,16,18,20}_* | xargs -i sh -c 'a=`basename {}`; ln -s {} $a'
rm *original*
cd ../iros2022-fisheye-tradition-test
ls -d ../../../../../../dataset/mot/mot17/{02,06,10,12,14,16,18,20}_original_* | xargs -i sh -c 'a=`basename {}`; ln -s {} $a'
cd ..
echo name > seqmaps/iros2022-fisheye-similar-test.txt | ls iros2022-fisheye-similar-test/ >> seqmaps/iros2022-fisheye-similar-test.txt
echo name > seqmaps/iros2022-fisheye-tradition-test.txt | ls iros2022-fisheye-tradition-test/ >> seqmaps/iros2022-fisheye-tradition-test.txt
```

## DeepSORT
Change working directory to `TGRMPT/tracking/deep_sort`.

Assume we have completed the detection and embedding feature extraction procedures, and saved results to `dataset/mot/mot17/${sequence}/embedding`. See [detection README](../detection/README.md) and [reid README](../reid/README.md) for detail.

To run DeepSORT on our TGRMPT test dataset, use the following commands:
```shell
bash deep_sort_app_wb.sh 5    # The parameter is the number of process, default is 10
bash deep_sort_app_hs.sh 5
```
To change hyper parameters of DeepSORT, see the shell scripts for detail. The above commands will save tracking results to `../eval/data/trackers/zjlab`.

To evaluate the above trackers, run
```shell
bash eval.sh
```

## Our TGRMPT
Change working directory to `TGRMPT/tracking/synthetic_deep_sort`.

Assume we have completed the detection and embedding feature extraction procedures, and saved results to `dataset/mot/mot17/${sequence}/embedding`. See [detection README](../detection/README.md) and [reid README](../reid/README.md) for detail.

To run our head-shoulder aided multi-person tracking method, use the following command:
```shell
bash deep_sort_app.py 5   # The parameter is the number of process, default is 10 
```
To change hyper parameters, see the shell script for detail.

To evaluate the above trackers, run
```shell
bash eval.sh
```
To evaluate using HOTA metric other than TGR_HOTA, modify `eval.sh` and change TGR_HOTA to HOTA.