#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
convert coco data

class x_center y_center width height

../datasets/coco128/images/im0.jpg  # image
../datasets/coco128/labels/im0.txt  # label
├── head_shoulder
│ └── labels
│ └── images
├── whole_body
│ └── labels
│ └── images
'''

import sys, os
from multiprocessing import Pool
from PIL import Image
import argparse
from functools import partial


parser = argparse.ArgumentParser(description="Convert TGRMPT dataset from mot to coco detection")
parser.add_argument("--num-process", default=8, help="Number of processes")
parser.add_argument("--data-path", required=True, help="The root path to TGRMPT dataset")
args = parser.parse_args()

dataset_path = args.data_path

num_process = args.num_process
# source
source_root = os.path.join(dataset_path, "mot/mot17")
link_img = False

# save path
save_root = os.path.join(dataset_path, 'detection')
images_root = os.path.join(save_root, 'images')

os.makedirs(os.path.join(images_root, "train"), exist_ok=True)
os.makedirs(os.path.join(images_root, "test"), exist_ok=True)


def one_process(path, label_file, labels_root, create_link):
    if int(path.split('_')[0]) % 2 == 0:
        subset = "test"
    else:
        subset = "train"
    label = [[int(float(x)) for x in l.strip().split(',')[:6]] for l in
             open(os.path.join(source_root, path, 'gt', label_file)).readlines()]
    label_info = {}
    for l in label:
        img_name = '%0.6d.jpg' % (l[0])
        if img_name not in label_info:
            label_info[img_name] = []
        label_info[img_name].append(l[2:])

    img_m = Image.open(os.path.join(source_root, path, 'img1', img_name))
    img_width = img_m.width
    img_height = img_m.height

    for k in label_info:
        if not os.path.exists(os.path.join(source_root, path, 'img1', k)):
            print('NO FIND IMAGE: ', os.path.join(source_root, path, 'img1', k))
            continue

        img_name = path + '_' + k
        if create_link:
            old_img_name = os.path.join(source_root, path, 'img1', k)
            new_img_name = os.path.join(images_root, subset, img_name)
            os.system('ln -s %s %s' % (old_img_name, new_img_name))

        out = []
        for b in label_info[k]:
            w = b[2] / img_width
            h = b[3] / img_height
            x = b[0] / img_width + w / 2
            y = b[1] / img_height + h / 2
            out.append(' '.join(map(str, [0, x, y, w, h])) + '\n')
        open(os.path.join(labels_root, subset, img_name.replace('jpg', 'txt')), 'w').writelines(out)


if __name__ == '__main__':
    path_list = [x for x in os.listdir(source_root)]
    print('num of paths: ', len(path_list))

    print("Create head shoulder detection")
    pool = Pool(num_process)
    label_file = "gt_hs.txt"
    labels_root = os.path.join(save_root, "head_shoulder", "labels")
    os.makedirs(os.path.join(labels_root, "train"), exist_ok=True)
    os.makedirs(os.path.join(labels_root, "test"), exist_ok=True)
    pool.map(partial(one_process, label_file=label_file, labels_root=labels_root, create_link=True), path_list)
    pool.close()
    pool.join()
    os.system('ln -s {} {}'.format(images_root, os.path.join(save_root, "head_shoulder/images")))

    print("Create whole body detection")
    pool = Pool(num_process)
    label_file = "gt_body.txt"
    labels_root = os.path.join(save_root, "whole_body", "labels")
    os.makedirs(os.path.join(labels_root, "train"), exist_ok=True)
    os.makedirs(os.path.join(labels_root, "test"), exist_ok=True)
    pool.map(partial(one_process, label_file=label_file, labels_root=labels_root, create_link=False), path_list)
    pool.close()
    pool.join()
    os.system('ln -s {} {}'.format(images_root, os.path.join(save_root, "whole_body/images")))

