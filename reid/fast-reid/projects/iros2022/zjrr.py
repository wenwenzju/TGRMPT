#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
user: hushunda
data: 2021/7/12
time: 4:39pm
email: hushunda@zhejianglab.com
'''

import os
from glob import glob
import re

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

__all__ = ['ZJRR', ]


@DATASET_REGISTRY.register()
class ZJRR(ImageDataset):
    """
    """
    dataset_dir = ""
    dataset_name = "zjrr"
    '''
    默认结构
    |--datasets
    |--|--images
    |--|--index_txt
    
    '''

    def __init__(self, root='', **kwargs):
        self.index_prefix = kwargs['index_prefix'] # txt_root
        self.image_root = kwargs["image_root"]
        txt_path = [self.index_prefix+x for x in ['_train.txt','_query.txt','_gallery.txt']]
        self.check_before_run(txt_path)

        train, query, gallery = self.process_train(txt_path)
        query, gallery = self.map_pid_camid(query, gallery)

        super().__init__(train, query, gallery, **kwargs)

    def process_train(self, txt_path):
        data = []
        for txt in txt_path:
            d = []
            for line in open(txt).readlines():
                line = line.strip()
                img_path = os.path.join(self.image_root,line+'.jpg')
                pid = re.search(r'\d{2,3}_\d{2}', line).group()
                line = line.split('/')
                pid = self.dataset_name + "_" + pid
                line = '_'.join(line[-1].split('_')[2:6])
                camid = self.dataset_name + "_" + line
                d.append((img_path, pid, camid))
            data.append(d)
        return data

    def map_pid_camid(self, query, gallery):
        pids = set([x[1] for x in query+gallery])
        cams = set([x[2] for x in query+gallery])
        pid_dict = dict([(p, i) for i, p in enumerate(sorted(pids))])
        cam_dict = dict([(p, i) for i, p in enumerate(sorted(cams))])
        # query = [(x[0], pid_dict[x[1]], cam_dict[x[2]]) for i,x in enumerate(query)]
        # gallery = [(x[0], pid_dict[x[1]], cam_dict[x[2]]) for i,x in enumerate(gallery)]
        query = [(x[0], pid_dict[x[1]], 0) for i,x in enumerate(query)]
        gallery = [(x[0], pid_dict[x[1]], 1) for i,x in enumerate(gallery)]
        return query, gallery
