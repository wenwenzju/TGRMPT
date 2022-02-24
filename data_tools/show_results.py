#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
user: hushunda
data: 2022/1/7
time: 下午6:48
email: hushunda@zhejianglab.com
'''

import cv2
import os

def show_detection():
    txt_file = '/home/zjrobot/TEMP/temp/gt_hs.txt'
    img_root = '/home/zjrobot/TEMP/temp/'
    data = [[x for x in l.strip().split(',')] for l in open(txt_file).readlines()]
    info = {}
    for l in data:
        frame = '%0.6d.jpg'%int(l[0])
        d = [int(x) for x in l[1:6]]
        if frame not in info:
            info[frame]=[d]
        else:
            info[frame].append(d)


    for img_name in info.keys():
        img = cv2.imread(os.path.join(img_root, img_name))
        if img_name in info:
            for bb in info[img_name]:
                bb[1] = bb[1]-bb[3]//2
                bb[2] = bb[2]-bb[4]//2
                bb[3] += bb[1]
                bb[4] += bb[2]
                cv2.rectangle(img, (bb[1], bb[2]), (bb[3], bb[4]), (0, 255, 0))

        cv2.imshow('img', img)
        cv2.waitKey()

if __name__ == '__main__':
    show_detection()
