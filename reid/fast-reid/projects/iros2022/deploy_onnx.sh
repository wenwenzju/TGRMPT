log_dir=logs/bot_r18_train_on_iros2022_fisheye_whole_body
python tools/onnx_export.py --config-file configs/whole_body.yml --name whole_body \
	--batch-size 10 --output ${log_dir} --opts MODEL.WEIGHTS ${log_dir}/model_best.pth

log_dir=logs/bot_r18_train_on_iros2022_fisheye_head_shoulder
python tools/onnx_export.py --config-file configs/head_shoulder.yml --name head_shoulder \
	--batch-size 10 --output ${log_dir} --opts MODEL.WEIGHTS ${log_dir}/model_best.pth
