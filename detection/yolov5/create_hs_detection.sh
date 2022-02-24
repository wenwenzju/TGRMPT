
img_root='/data/dataset/iros2022/mot/mot17'
gpu_ids=(0 1 2 3)
for sub_path in `ls $img_root`;
  do
#    for x in ${gpu_ids[*]};
#    do{
      mkdir ${img_root}/${sub_path}/detection
      python detect.py --weights runs/hs/exp16/weights/best.pt --source ${img_root}/${sub_path}/img1 --device 0 --class 0 --name hs
#      } &
#    done
#    wait
done;