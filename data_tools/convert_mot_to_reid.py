#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
crop headshoulder images and index from mot fish data
'''
import cv2
import sys, os
import numpy as np
from pycocotools.coco import COCO
from multiprocessing import Pool


source_root = '/data/dataset/iros2022/mot/mot17'
save_root = '/data/dataset/iros2022/reid/hs_images'

index_path = os.path.join(save_root,'../hs_index_txt')
label_file = 'gt_hs.txt'
num_process = 32
max_id = 10 # filter images with id>max_id

def one_process(path):
    label = [[int(float(x)) for x in l.strip().split(',')[:6]] for l in open(os.path.join(source_root,path,'gt',label_file)).readlines()]
    label_info = {}
    for l in label:
        img_name ='%0.6d.jpg'%(l[0])
        if img_name not in label_info:
            label_info[img_name] =[]
        label_info[img_name].append(l[1:])

    for k in label_info:
        if not os.path.exists(os.path.join(source_root,path,'img1',k)):
            print('NO FIND IMAGE: ',os.path.join(source_root,path,'img1',k))
            continue
        img = cv2.imread(os.path.join(source_root,path,'img1',k))
        group = path.split('_')[0]
        for b in label_info[k]:
            if b[0]>max_id:continue
            save_sub_folder = os.path.join(save_root,group+'_%0.2d'%(b[0]))
            os.makedirs(save_sub_folder,exist_ok=True)
            sub_name =path[3:].split('_')
            sub_name.insert(2,'nvidia3')
            sub_name = '_'.join(sub_name)
            save_name = os.path.join(save_sub_folder,sub_name+'_'+'%0.4d.jpg'%(int(k.split('.')[0])-1))
            b[1]=max(0,b[1])
            b[2]=max(0,b[2])
            x1,y1,x2,y2 = b[1],b[2],b[3]+b[1],b[4]+b[2]
            if x2<=x1 or y2<=y1:
                print(x1,y1,x2,y2,'source name: ',path+'_'+k,'save name:',save_name)
                continue
            crop_img = img[y1:y2,x1:x2]
            try:
                cv2.imwrite(save_name,crop_img)
            except:
                print(x1, y1, x2, y2, 'source name: ', path + '_' + k, 'save name:', save_name)

def create_index_txt():
    pass

def verify_index():
    count = 0
    index_path = '/data/dataset/iros2022/reid/index_txt'
    for file in os.listdir(index_path):
        if not 'fish' in file:continue
        file_name = [os.path.join(save_root, l.strip()+'.jpg') for l in open(os.path.join(index_path,file)).readlines()]
        for f in file_name:
            if not os.path.exists(f):
                count+=1
                print(f)
    print(count)

if __name__ == '__main__':

    pool = Pool(num_process)

    path_list = [x for x in os.listdir(source_root) if 'fish' in x]
    print('num of paths: ',len(path_list))

    pool.map(one_process, path_list)
    pool.close()
    pool.join()

    verify_index()