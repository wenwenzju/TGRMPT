#!/bin/bash

# Detection confidence threshold. Disregard all detections that have a confidence lower than this value.
min_confidence=0.8

# Threshold on the detection bounding box height. Detections with height smaller than this value are disregarded.
min_detection_height=0

# Gating threshold for cosine distance metric (object appearance).
max_cosine_distances=(0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9)
# max_cosine_distances=(0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9)
# max_cosine_distances=(0.2)

# Maximum size of the appearance descriptors gallery. If None, no budget is enforced.
nn_budget=50

# Maximum number of missed misses before a track is deleted. -1 means infinite
max_ages=(-1)
# max_ages=(-1)

gt_base=../eval/data/gt/zjlab
tracker_base=../eval/data/trackers/zjlab
tracker_method=deep_sort_wb

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
proc_num=10
if [ $# -ge 1 ]; then
  proc_num=$1
fi

for ((n=1; n<=${proc_num}; n++))
do
  echo >&1000
done

for max_cosine_distance in ${max_cosine_distances[*]}
do
  for max_age in ${max_ages[*]}
  do
    tracker_name=${tracker_method}_distance${max_cosine_distance}_budget${nn_budget}_age${max_age}

    if [[ ! -d ${tracker_base}/iros2022-fisheye-tradition-test/${tracker_name}/data ]]; then
      mkdir -p ${tracker_base}/iros2022-fisheye-tradition-test/${tracker_name}/data
    fi

    if [[ ! -d ${tracker_base}/iros2022-fisheye-similar-test/${tracker_name}/data ]]; then
      mkdir -p ${tracker_base}/iros2022-fisheye-similar-test/${tracker_name}/data
    fi

    for video in ${videos[*]}
    do
      read -u1000
      {
        echo "Process ${video} using ${tracker_name}"
        if [[ ${video} =~ "original" ]]; then
          cloth_type=original
        else
          cloth_type=similar
        fi
        python deep_sort_app.py --sequence_dir ${gt_base}/iros2022-fisheye-${cloth_type}-test/${video} \
            --detection_file ${gt_base}/iros2022-fisheye-${cloth_type}-test/${video}/embedding/embedding_wb.pkl \
            --output_file ${tracker_base}/iros2022-fisheye-${cloth_type}-test/${tracker_name}/data/${video}.txt \
            --min_confidence ${min_confidence} --min_detection_height ${min_detection_height} --max_cosine_distance ${max_cosine_distance} \
            --nn_budget ${nn_budget} --max_age ${max_age} --display False

        echo >&1000
      }&

    done

  done
done

wait

exec 1000>&-
exec 1000<&-

echo "Done."
