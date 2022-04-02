cd ../eval
gt_folder=data/gt/zjlab
tracker_folder=data/trackers/zjlab
benchmarks=(iros2022-fisheye-similar iros2022-fisheye-tradition)
split_to_eval=test

for benchmark in ${benchmarks[*]}
do
  seqs_path=${gt_folder}/${benchmark}-${split_to_eval}

  find -L ${seqs_path} -maxdepth 2 -type d -name "gt" -exec cp {}/gt_body.txt {}/gt.txt \;
  trackers=`cd ${tracker_folder}/${benchmark}-${split_to_eval}; ls -d deep_sort_concat_distance*`

  python scripts/run_mot_challenge.py --USE_PARALLEL True --NUM_PARALLEL_CORES 40 \
    --GT_FOLDER ${gt_folder} --TRACKERS_FOLDER ${tracker_folder} --TRACKERS_TO_EVAL ${trackers}\
    --BENCHMARK ${benchmark} --SPLIT_TO_EVAL ${split_to_eval} --METRICS TGR_HOTA CLEAR Identity

done
