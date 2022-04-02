CUDA_VISIBLE_DEVICES='0' python train.py --local_rank=0 --noval --weights pretrained_model/yolov5s.pt --batch-size 32 --workers 8 --data data/wb_coco.yaml --project runs/wb/ --cfg yolov5s.yaml
