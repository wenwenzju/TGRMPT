export CUDA_VISIBLE_DEVICES=$1
config=configs/whole_body.yml
log_dir=bot_r18_extract_mot_features_whole_body
model_dir=bot_r18_train_on_iros2022_fisheye_whole_body/model_best.pth

videos=(
  02_black_black_fisheye_head_front
  02_blue_black_fisheye_head_front
  02_color_gray_fisheye_head_front
  02_original_black_fisheye_head_front
  02_suit_black_fisheye_head_front
  06_color_gray_fisheye_head_front
  06_gray_gray_fisheye_head_front
  06_original_black_fisheye_head_front
  06_red_gray_fisheye_head_front
  06_suit_black_fisheye_head_front
  10_black_black_fisheye_head_front
  10_blue_black_fisheye_head_front
  10_gray_gray_fisheye_head_front
  10_original_black_fisheye_head_front
  10_red_gray_fisheye_head_front
  12_blue_black_fisheye_head_front
  12_color_gray_fisheye_head_front
  12_original_black_fisheye_head_front
  12_suit_black_fisheye_head_front
  12_white_black_fisheye_head_front
  14_color_gray_fisheye_head_front
  14_gray_gray_fisheye_head_front
  14_original_black_fisheye_head_front
  14_red_gray_fisheye_head_front
  14_suit_black_fisheye_head_front
  16_blue_black_fisheye_head_front
  16_color_black_fisheye_head_front
  16_original_black_fisheye_head_front
  16_red_gray_fisheye_head_front
  16_suit_black_fisheye_head_front
  18_blue_black_fisheye_head_front
  18_gray_gray_fisheye_head_front
  18_original_black_fisheye_head_front
  18_red_gray_fisheye_head_front
  18_suit_black_fisheye_head_front
  20_gray_gray_fisheye_head_front
  20_original_black_fisheye_head_front
  20_red_gray_fisheye_head_front
  20_suit_black_fisheye_head_front
  20_white_black_fisheye_head_front
)

trap "exec 1000>&-;exec 1000<&-;exit 0" 2

mkfifo tracker_fifo
exec 1000<>tracker_fifo
rm -rf tracker_fifo
proc_num=5
if [ $# -ge 2 ]; then
  proc_num=$2
fi

for ((n=1; n<=${proc_num}; n++))
do
  echo >&1000
done

for video in ${videos[*]}
do
  read -u1000
  {
    echo "Process ${video}"
    seq=../../../../dataset/mot/mot17/${video}
    python extract_mot_features.py --config-file ${config} --mot ${seq} \
        --detection ${seq}/detection/det_wb.txt --output ${seq}/embedding/embedding_wb.pkl \
        --opts MODEL.WEIGHTS logs/${model_dir} OUTPUT_DIR logs/${log_dir}
    echo >&1000
  }&
done

wait
exec 1000>&-
exec 1000<&-

echo "Done."
