img_root='../../dataset/mot/mot17'
for sub_path in `ls -d $img_root/{02,06,10,12,14,16,18,20}_*`;
  do
      mkdir ${sub_path}/detection
      python detect.py --weights runs/wb/exp/weights/best.pt --source ${sub_path}/img1 --device 0 --class 0 --name wb --nosave
done;
