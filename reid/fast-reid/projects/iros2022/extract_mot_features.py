# encoding: utf-8
"""
@author:  Wang Wen
@contact: wangwen@zhejianglab.com
"""

import logging
import os
import argparse
import sys
import cv2
import torch
import torchvision.transforms as T
from tqdm import tqdm
import pickle
from PIL import Image
import multiprocessing
from functools import partial

import numpy as np

sys.path.append('../..')

from fastreid.config import get_cfg
from fastreid.modeling.meta_arch import build_model
from fastreid.utils.file_io import PathManager
from fastreid.utils.checkpoint import Checkpointer
from fastreid.utils.logger import setup_logger
from fastreid.data.transforms import build_transforms
from fastreid.data.transforms.transforms import ToTensor

setup_logger(name="fastreid")
logger = logging.getLogger("fastreid.extract_features")


def setup_cfg(args):
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Extract REID features of MOT videos")

    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--mot",
        required=True,
        type=str,
        help="Path to MOT video. This path should contain img1 subfolder."
    )
    parser.add_argument(
        "--detection",
        required=True,
        metavar="FILE",
        help='Path to detection file in MOTChallenge format.'
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Path of file to save."
    )
    parser.add_argument(
        '--batch-size',
        default=150,
        type=int,
        help="the maximum batch size to extract features"
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def load_batches(bbxes_batch, img_path, transforms, total):
    if len(bbxes_batch) == 0:
        return []

    batch_imgs, batch_res = [], []
    for bbxes in tqdm(bbxes_batch):
        frame_id = bbxes[0][0]
        img = cv2.imread(os.path.join(img_path, "%06d" % int(frame_id) + ".jpg"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for det in bbxes:
            x1, y1, w, h = det[2:6]
            x2 = x1 + w
            y2 = y1 + h
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            if x1 < 0: x1 = 0
            elif x1 >= img.shape[1]: x1 = img.shape[1] - 1
            if x2 < 0: x2 = 0
            elif x2 >= img.shape[1]: x2 = img.shape[1] - 1

            if y1 < 0: y1 = 0
            elif y1 >= img.shape[0]: y1 = img.shape[0] - 1
            if y2 < 0: y2 = 0
            elif y2 >= img.shape[0]: y2 = img.shape[0] - 1

            one_person = img[y1:y2, x1:x2, :]
            one_person = cv2.resize(one_person, (128, 256), interpolation=cv2.INTER_CUBIC)
            # one_person = transforms(Image.fromarray(one_person))

            # batch_imgs.append(one_person.unsqueeze(0))
            batch_imgs.append(one_person)
            batch_res.append(det.tolist())

        # print("Done {}/{}".format(int(frame_id), total))
    return batch_imgs, batch_res


def extract(args, model, cfg):
    img_path = os.path.join(args.mot, "img1")
    if not os.path.exists(img_path):
        logger.critical("{} doesn't contain subfolder img1".format(args.mot))
        return

    # load detections
    if not os.path.exists(args.detection):
        logger.critical("Detection file {} doesn't exist".format(args.detection))
        return
    detections = np.loadtxt(args.detection, delimiter=',')
    fid_dets = {}
    for det in detections:
        fid_dets.setdefault(det[0], []).append(det)
    fids = list(fid_dets)
    dets = [fid_dets[fid] for fid in fids]
    # dets = dets[:200]
    dets_batch = []
    for i in range(len(dets) // 1000 + 1):
        dets_batch.append(dets[i*1000:(i+1)*1000])
    # dets = dets[:200]

    # transforms = build_transforms(cfg, is_train=False)
    transforms = T.Compose([ToTensor()])

    pool = multiprocessing.Pool(5)
    batches = pool.map(partial(load_batches, img_path=img_path, transforms=transforms, total=len(dets)), dets_batch)
    print("Done load detection. Start extracting reid features...")

    all_imgs, all_dets = [], []
    for box_per_img, res_per_img in batches:
        all_imgs.extend(box_per_img)
        all_dets.extend(res_per_img)
    pre_frame_id = -1
    batch_imgs, batch_res, all_res = [], [], []
    # img = []
    # for det in tqdm(detections):
    for one_person, det in tqdm(zip(all_imgs, all_dets), total=len(all_imgs)):
        # cur_frame_id = det[0]
        # if cur_frame_id != pre_frame_id:
        #     img = cv2.imread(os.path.join(img_path, "%06d" % int(cur_frame_id)+".jpg"))
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #     pre_frame_id = cur_frame_id
        # x1, y1, w, h = det[2:6]
        # x2 = x1 + w
        # y2 = y1 + h
        # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # if x1 < 0: x1 = 0
        # elif x1 >= img.shape[1]: x1 = img.shape[1] - 1
        # if x2 < 0: x2 = 0
        # elif x2 >= img.shape[1]: x2 = img.shape[1] - 1

        # if y1 < 0: y1 = 0
        # elif y1 >= img.shape[0]: y1 = img.shape[0] - 1
        # if y2 < 0: y2 = 0
        # elif y2 >= img.shape[0]: y2 = img.shape[0] - 1

        # one_person = img[y1:y2, x1:x2, :]
        one_person = transforms(Image.fromarray(one_person))

        batch_imgs.append(one_person.unsqueeze(0))
        batch_res.append(det)
        if len(batch_imgs) >= args.batch_size:
            inputs = {"images": torch.cat(batch_imgs[:args.batch_size]).to(model.device)}
            outputs = model(inputs).cpu().detach().numpy()
            for i in range(args.batch_size):
                batch_res[i].extend(outputs[i].tolist())
            all_res.extend(batch_res[:args.batch_size])
            batch_imgs = batch_imgs[args.batch_size:]
            batch_res = batch_res[args.batch_size:]

    if len(batch_imgs) > 0:
        inputs = {"images": torch.cat(batch_imgs).to(model.device)}
        outputs = model(inputs).cpu().detach().numpy()
        for i in range(len(batch_res)):
            batch_res[i].extend(outputs[i].tolist())
        all_res.extend(batch_res)

    return all_res


if __name__ == '__main__':
    args = get_parser().parse_args()
    output = os.path.abspath(args.output)
    if os.path.exists(output):
        print("Already done for {}. Exit.".format(output))
        sys.exit(0)
    cfg = setup_cfg(args)

    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    model = build_model(cfg)
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()
    logger.info(model)

    res = extract(args, model, cfg)

    if not os.path.exists(os.path.dirname(output)):
        os.makedirs(os.path.dirname(output))
    with open(output, 'wb') as f:
        pickle.dump(res, f)

    logger.info("Features has already saved to {}!".format(args.output))
