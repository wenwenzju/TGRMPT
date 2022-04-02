# Deploy
Change working directory to `TGRMPT/deploy`.

To deploy our method on board of a robot using TensorRT, first, deploy whole body and head shoulder detection models (see [detection README](../detection/README.md)), 
and copy the generated engine models into the folder `detection/weights`, and be sure to rename them to `head_shoulder.engine` and `whole_body.engine`.
Second, deploy whole body and head shoulder ReID models (see [reid README](../reid/README.md)), and copy the generated engine models into the folder `embedding/weights`, 
and be sure to rename them to `head_shoulder.engine` and `whole_body.engine`.

To run our method on a sequence of images, e.g., `../dataset/mot/mot17/02_original_black_fisheye_head_front`, run
```shell
python deep_sort_app.py --sequence_dir ../dataset/mot/mot17/02_original_black_fisheye_head_front/img1
```
Run `python deep_sort_app.py --help` to see more options.