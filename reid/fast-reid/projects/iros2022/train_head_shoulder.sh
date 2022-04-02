export CUDA_VISIBLE_DEVICES=$1
config=configs/head_shoulder.yml
log_dir=bot_r18_train_on_iros2022_fisheye_head_shoulder
python train_net.py --config-file ${config} --num-gpus 1 SOLVER.IMS_PER_BATCH 64 OUTPUT_DIR logs/${log_dir}

python train_net.py --config-file ${config} --eval-only MODEL.WEIGHTS logs/${log_dir}/model_best.pth \
        OUTPUT_DIR logs/${log_dir}
