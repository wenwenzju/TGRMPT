import os
import sys
import numpy as np
import json
import cv2
'''
将数据集划分成训练集和测试集。
'''

mode = 'body' # body or hs
if len(sys.argv)>1:
    mode = sys.argv[1]
print("mode: ",mode)

# Use the same script for MOT16
DATA_PATH = '/data/dataset/iros2022/mot/'
OUT_PATH = os.path.join(DATA_PATH, mode+'_annotations')
gt_file = 'gt_%s.txt'%(mode)
SPLITS = ['train_'+mode, 'test_'+mode]
HALF_VIDEO = True
CREATE_SPLITTED_ANN = True


if __name__ == '__main__':

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    for split in SPLITS:
        data_path = os.path.join(DATA_PATH,'mot17')
        out_path = os.path.join(OUT_PATH, '{}.json'.format(split))
        out = {'images': [], 'annotations': [], 'videos': [],
               'categories': [{'id': 1, 'name': 'pedestrian'}]}
        seqs = os.listdir(data_path)
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        tid_curr = 0
        tid_last = -1
        train_seqs = ['%0.2d'%x for x in range(1,21,2)]
        for seq in sorted(seqs):
            if 'fish' not in seq:continue
            train_ok = ('train' in split) and (seq.split('_')[0] in train_seqs)
            test_ok = ('test' in split) and (seq.split('_')[0] not in train_seqs)
            if not (train_ok or test_ok):
                continue
            video_cnt += 1  # video sequence number.
            out['videos'].append({'id': video_cnt, 'file_name': seq})
            seq_path = os.path.join(data_path, seq)
            img_path = os.path.join(seq_path, 'img1')
            ann_path = os.path.join(seq_path, 'gt', gt_file)
            images = os.listdir(img_path)
            num_images = len([image for image in images if 'jpg' in image])  # half and half
            img = cv2.imread(os.path.join(data_path, '{}/img1/{:06d}.jpg'.format(seq,  1)))
            for i in range(num_images):
                height, width = img.shape[:2]
                image_info = {'file_name': '{}/img1/{:06d}.jpg'.format(seq, i + 1),  # image name.
                              'id': image_cnt + i + 1,  # image number in the entire training set.
                              'frame_id': i + 1 ,  # image number in the video sequence, starting from 1.
                              'prev_image_id': image_cnt + i if i > 0 else -1,  # image number in the entire training set.
                              'next_image_id': image_cnt + i + 2 if i < num_images - 1 else -1,
                              'video_id': video_cnt,
                              'height': height, 'width': width}
                out['images'].append(image_info)
            print('{}: {} images'.format(seq, num_images))
            anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
            anns_out = np.array([anns[i] for i in range(anns.shape[0])], np.float32)
            gt_out = os.path.join(seq_path, 'gt/gt_{}.txt'.format(split))
            fout = open(gt_out, 'w')
            for o in anns_out:
                fout.write('{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:.6f}\n'.format(
                            int(o[0]), int(o[1]), int(o[2]), int(o[3]), int(o[4]), int(o[5]),
                            int(o[6]), int(o[7]), o[8]))
            fout.close()

            print('{} ann images'.format(int(anns[:, 0].max())))
            for i in range(anns.shape[0]):
                frame_id = int(anns[i][0])
                track_id = int(anns[i][1])
                cat_id = int(anns[i][7])
                ann_cnt += 1
                category_id = 1
                ann = {'id': ann_cnt,
                       'category_id': category_id,
                       'image_id': image_cnt + frame_id,
                       'track_id': tid_curr,
                       'bbox': anns[i][2:6].tolist(),
                       'conf': float(anns[i][6]),
                       'iscrowd': 0,
                       'area': float(anns[i][4] * anns[i][5])}
                out['annotations'].append(ann)
            image_cnt += num_images
            print(tid_curr, tid_last)
        print('loaded {} for {} images and {} samples'.format(split, len(out['images']), len(out['annotations'])))
        json.dump(out, open(out_path, 'w'),indent=4)