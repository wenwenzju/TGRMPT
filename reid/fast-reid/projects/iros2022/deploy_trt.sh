onnx_name=whole_body
trt_name=${onnx_name}
log_dir=logs/bot_r18_train_on_iros2022_fisheye_${onnx_name}
python ../../tools/deploy/trt_export.py --name ${trt_name} --output ${log_dir} --mode fp32 --batch-size 10 --height 256 --width 128 \
       --onnx-model ${log_dir}/${onnx_name}.onnx

onnx_name=head_shoulder
trt_name=${onnx_name}
log_dir=logs/bot_r18_train_on_iros2022_fisheye_${onnx_name}
python ../../tools/deploy/trt_export.py --name ${trt_name} --output ${log_dir} --mode fp32 --batch-size 10 --height 256 --width 128 \
       --onnx-model ${log_dir}/${onnx_name}.onnx
