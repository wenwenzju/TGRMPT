#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
convert coco data

class x_center y_center width height

../datasets/coco128/images/im0.jpg  # image
../datasets/coco128/labels/im0.txt  # label
├── headshoulder
│ └── labels
├── images
├── index_txt
├── labels
└── wholebody
    └── labels
'''

import sys, os
from multiprocessing import Pool
from PIL import Image

num_process = 32
# source
source_root = '/data/dataset/iros2022/mot/mot17'
link_img = False

# save path
save_root = '/data/dataset/iros2022/detection'
images_root = os.path.join(save_root,'images')
index_path = os.path.join(save_root,'index_txt')

label_dict={'gt_body.txt':'wholebody','gt_hs.txt':'headshoulder'}
label_file = 'gt_body.txt' # gt_body
labels_root = os.path.join(save_root,label_dict[label_file],'labels')

os.makedirs(index_path,exist_ok=True)
os.makedirs(images_root,exist_ok=True)
os.makedirs(labels_root,exist_ok=True)


def one_process(path):
    label = [[int(float(x)) for x in l.strip().split(',')[:6]] for l in open(os.path.join(source_root,path,'gt',label_file)).readlines()]
    label_info = {}
    for l in label:
        img_name ='%0.6d.jpg'%(l[0])
        if img_name not in label_info:
            label_info[img_name] =[]
        label_info[img_name].append(l[2:])

    img_m = Image.open(os.path.join(source_root,path,'img1',img_name))
    img_width = img_m.width
    img_height = img_m.height

    for k in label_info:
        if not os.path.exists(os.path.join(source_root,path,'img1',k)):
            print('NO FIND IMAGE: ',os.path.join(source_root,path,'img1',k))
            continue

        old_img_name = os.path.join(source_root,path,'img1',k)
        img_name = path+'_'+k
        new_img_name = os.path.join(images_root,img_name)
        if link_img:
            os.system('ln -s %s %s'%(old_img_name,new_img_name))

        out = []
        for b in label_info[k]:
            w = b[2] / img_width
            h = b[3] / img_height
            x = b[0] / img_width + w / 2
            y = b[1] / img_height + h / 2
            out.append(' '.join(map(str, [0, x, y, w, h])) + '\n')
        open(os.path.join(labels_root, img_name.replace('jpg', 'txt')), 'w').writelines(out)


def create_index_txt():
    train_index = []
    test_index = []

    train_group = ['%0.2d'%(x) for x in range(1,21,2)]
    for img_name in os.listdir(images_root):
        group = img_name.split('_')[0]
        if group in train_group:
            train_index.append(img_name+'\n')
        else:
            test_index.append(img_name + '\n')

    open(os.path.join(index_path, 'train.txt'), 'w').writelines(train_index)
    open(os.path.join(index_path, 'test.txt'), 'w').writelines(test_index)



if __name__ == '__main__':

    pool = Pool(num_process)

    path_list = [x for x in os.listdir(source_root) if 'fish' in x]
    print('num of paths: ',len(path_list))

    pool.map(one_process, path_list)
    pool.close()
    pool.join()



