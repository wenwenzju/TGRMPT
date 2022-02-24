import os
import json
from PIL import Image
import numpy as np
from multiprocessing import Pool
from loguru import logger

'''
完成头肩和全身框同步，并且转化成mot格式。
mot17:标注格式：
<frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<x>,<y>,<z>
'''


hs_root= '/data/dataset/hs_mot_label/yolov5s'
body_root = '/data/dataset/mot'
save_root = '/data/dataset/iros2022/mot/mot17'

num_process = 32
suffix = '_manual_check_consistent_filter.json'
out_hs_label = True
out_body_label = True
out_images = False
out_seqinfo = True


@logger.catch
def one_process(path_info):
    g, t, cam, img_sub_path = path_info
    body_label = json.load(open(os.path.join(body_root, g, t, cam)))
    out_hs_label = []
    out_body_label = []
    # 保存信息
    imExt = '.jpg'
    frameRate = 25
    imDir = 'img1'
    name = g + '_' + '_'.join(t.split('-')[-1].split('_')[:-1]) + '_' + img_sub_path

    out_save_root = os.path.join(save_root, name)
    os.makedirs(out_save_root, exist_ok=True)
    os.makedirs(os.path.join(out_save_root, imDir), exist_ok=True)

    # 初始化
    img_names = sorted(body_label.keys())
    last_frame = 0
    # last_t = float(img_names[0].split('.')[0])
    imWidth, imHeight = 0, 0

    img_path = os.path.join(body_root, g, t, img_sub_path)
    img_name_list = sorted(os.listdir(img_path))
    for b_key in img_name_list:
        img_name = os.path.join(body_root, g, t, img_sub_path, b_key)

        # 计算帧数
        cur_t = float(b_key.split('.')[0])
        delta = 1  # time2frame(cur_t,last_t,frameRate)
        cur_frame = last_frame + delta

        # 计算检测框
        if b_key in img_names:
            img_m = Image.open(img_name)
            imWidth = img_m.width
            imHeight = img_m.height
            hs_label_file = os.path.join(hs_root, g, t, img_sub_path, b_key.replace('jpg', 'txt'))
            if os.path.exists(hs_label_file):
                hs_label = [[float(x) for x in l.strip().split(' ')] for l in open(hs_label_file).readlines()]
            else:
                hs_label = [[]]
            # body label
            bd_label = body_label[b_key]
            # 过滤ID小于1的
            bd_label = [x for x in bd_label if int(x[4])>0]
            bd_label = np.maximum(0,np.array(bd_label)).tolist()

            out_body_label.extend(gen_body_label(bd_label, cur_frame))
            hs_gt = syn_label(bd_label, hs_label, (imWidth, imHeight))
            for gt in hs_gt:
                out_hs_label.append(','.join(map(str, ['%0.6d' % cur_frame] + gt)) + '\n')

        # 生成图像链接
        if out_images:
            os.system('ln -s %s %s' % (img_name, os.path.join(out_save_root, imDir, '%0.6d' % cur_frame + imExt)))
        # 更新帧数
        last_frame = cur_frame
        last_t = cur_t

    # 保存label
    gt_root = os.path.join(out_save_root, 'gt')
    os.makedirs(gt_root, exist_ok=True)
    if out_hs_label:
        open(os.path.join(gt_root, 'gt_hs.txt'), 'w').writelines(out_hs_label)
    if out_body_label:
        open(os.path.join(gt_root, 'gt_body.txt'), 'w').writelines(out_body_label)

    # 保存信息 seqinfo.ini
    if out_seqinfo:
        with open(os.path.join(out_save_root, 'seqinfo.ini'), 'w') as f:
            data = "[Sequence]\n" \
                   "name=%s\n" \
                   "imDir=%s\n" \
                   "frameRate=%d\n" \
                   "seqLength=%d\n" \
                   "imWidth=%d\n" \
                   "imHeight=%d\n" \
                   "imExt=%s" % (name, imDir, frameRate, last_frame, imWidth, imHeight, imExt)
            f.writelines(data)

def syn_label(bl,hsl,wh):
    '''

    :param bl:（0~width）
    :param hsl:（0~1）
    :return: left，top，width，height
    '''
    def iou(ba,bb):
        '''

        :param ba: 一个框
        :param bb: n个框
        :return:
        '''
        bb = np.array(bb)
        x1 = np.maximum(bb[:,0],ba[0])
        y1 = np.maximum(bb[:,1],ba[1])
        x2 = np.minimum(bb[:,2],ba[2])
        y2 = np.minimum(bb[:,3],ba[3])
        inter = np.maximum(0,x2-x1)*np.maximum(0,y2-y1)

        un = (bb[:,2]-bb[:,0])*(bb[:,3]-bb[:,1])
        return inter/(un+1e-6)


    # 过滤掉置信度低的
    hsl = np.array(hsl)
    if len(hsl[0])==0:
        out = []
        for b in bl:
            bb = b[:4]
            ids = b[4]
            x1, y1, x2, y2 = bb
            if x2 - x1 < y2 - y1:
                w = x2 - x1
                gt_b = [x1, y1, w, w]
            else:
                gt_b = [x1, y1, x2 - x1, y2 - y1]
            gt_b = np.array(gt_b).astype(np.int).tolist()
            out.append([ids, *gt_b, 1, 1, 1,1])
        return out
    hsl = hsl[hsl[:,-1]>0.3]
    hsl = hsl[:,1:5]*np.array([wh[0],wh[1],wh[0],wh[1]])
    hsl[:,0] = hsl[:,0]-hsl[:,2]/2
    hsl[:,1] = hsl[:,1]-hsl[:,3]/2
    hsl[:,2] = hsl[:,2]+hsl[:,0]
    hsl[:,3] = hsl[:,3]+hsl[:,1]


    out = []
    for b in bl:
        bb = b[:4]
        ids = b[4]
        if len(hsl)==0:
            x1,y1,x2,y2 =bb
            if x2-x1<y2-y1:
                w = x2-x1
                gt_b = [x1,y1,w,w]
            else:
                gt_b = [x1, y1, x2-x1, y2-y1]
            gt_b = np.array(gt_b).astype(np.int).tolist()
            out.append([ids,*gt_b,1,1,1,1])
            continue
        ious = iou(bb,hsl)
        if ious.max()>0.6:
            miou_ind = np.argmax(ious)
            gt_b = hsl[miou_ind]
            gt_b[2] = gt_b[2]-gt_b[0]
            gt_b[3] = gt_b[3]-gt_b[1]
            hsl = np.delete(hsl,miou_ind,axis=0)
        else:
            x1,y1,x2,y2 =bb
            if x2-x1<y2-y1:
                w = x2-x1
                gt_b = [x1,y1,w,w]
            else:
                gt_b = [x1, y1, x2-x1, y2-y1]
        gt_b = np.array(gt_b).astype(np.int).tolist()
        out.append([ids,*gt_b,1,1,1,1])
    return out

# def load_seq_label(body_label, interval = 1):
#     '''将载入的label，按照时间切分成几份'''
#     times = sorted([float(x.split('.'))/1e9 for x in body_label.keys()])
#
#     return label

def gen_body_label(label,cur_frame):
    out = []
    for b in label:
        t = [b[-1],int(b[0]),int(b[1]),int(b[2]-b[0]),int(b[3]-b[1]),1,1,1,1]
        out.append(','.join(map(str,['%0.6d'%cur_frame]+t))+'\n')
    return out

def time2frame(t,lt,fps=25):
    '''

    :param t: 当前时间
    :param lt: 上一帧时间
    :param fps: 频率
    :return: 增加的frame
    '''
    td=((t-lt)/1e9)*fps
    return max(1,round(td))

if __name__ == '__main__':

    pool = Pool(num_process)

    path_list = []
    for g in os.listdir(body_root):
        for t in os.listdir(os.path.join(body_root,g)):
            for cam in os.listdir(os.path.join(body_root, g, t)):
                if not cam.endswith(suffix):continue
                img_sub_path = cam.replace(suffix, '')
                path_list.append([g, t, cam, img_sub_path])

    pool.map(one_process, path_list)
    pool.close()
    pool.join()
