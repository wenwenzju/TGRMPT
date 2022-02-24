"""
Combine whole body and head shoulder detection results.
1. Match whole body and head shoulder detections in each frame using Hungarian matching.
2. For single detection in whole body, infer head shoulder detection.
3. For single detection in head shoulder, infer whole body detection.
"""
import os
from tqdm import tqdm
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment


def load_detection(path):
    ret = {}
    with open(path, 'r') as f:
        for l in f.readlines():
            frame_id, det_id, x, y, w, h, conf, X, Y, Z = l.strip().split(',')
            ret.setdefault(frame_id, []).append([int(det_id), int(x), int(y), int(w), int(h), float(conf), X, Y, Z])
        return ret


def save_detection(path, dets):
    with open(path, 'w') as f:
        frame_ids = list(dets)
        frame_ids.sort()
        for frame_id in frame_ids:
            for box in dets[frame_id]:
                f.write("%s,%d,%d,%d,%d,%d,%f,%s,%s,%s\n" % (frame_id, box[0], box[1], box[2], box[3], box[4],
                                                           box[5], box[6], box[7], box[8]))


def iou(bbox, candidates):
    """Computer intersection over candidates.

    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.

    Returns
    -------
    ndarray
        The intersection over candidates in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.

    """
    bbox = np.asarray(bbox, dtype=np.float32)
    candidates = np.asarray(candidates, dtype=np.float32)
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection / area_candidates


def main():
    video_path = '/media/wenwenzju/disk/dataset/zjlab/iros2022/mot/mot17'
    videos = os.listdir(video_path)
    width_ratio, height_ratio = 0.78, 0.31
    for video in tqdm(videos):
        det_wb = load_detection(os.path.join(video_path, video, "detection/gt_wb.txt"))
        det_hs = load_detection(os.path.join(video_path, video, "detection/gt_hs.txt"))
        wb_frames = set(det_wb.keys())
        hs_frames = set(det_hs.keys())
        inter_frames = wb_frames & hs_frames            # both detected by whole body and head shoulder
        diff_wb_frames = wb_frames - inter_frames       # only detected by whole body
        diff_hs_frames = hs_frames - inter_frames       # only detected by head shoulder

        for frame_id in inter_frames:
            wb_boxes = np.array(det_wb[frame_id])[:, 1:5]
            hs_boxes = np.array(det_hs[frame_id])[:, 1:5]
            cost_matrix = np.zeros((len(wb_boxes), len(hs_boxes)))
            for row in range(len(wb_boxes)):
                cost_matrix[row, :] = 1. - iou(wb_boxes[row], hs_boxes)
            row_indices, col_indices = linear_assignment(cost_matrix)
            matches = []
            unmatched_hs = [col for col in range(len(hs_boxes)) if col not in col_indices]
            unmatched_wb = [row for row in range(len(wb_boxes)) if row not in row_indices]
            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] > 0.4:
                    unmatched_wb.append(row)
                    unmatched_hs.append(col)
                else:
                    matches.append((row, col))
            det_id = 1
            for match in matches:
                wb_idx, hs_idx = match
                det_wb[frame_id][wb_idx][0] = det_id
                det_hs[frame_id][hs_idx][0] = det_id
                det_id += 1
            for wb_idx in unmatched_wb:
                x, y, w, h, conf, X, Y, Z = det_wb[frame_id][wb_idx][1:]
                w_hs = w * width_ratio
                h_hs = h * height_ratio
                det_wb[frame_id][wb_idx][0] = det_id
                det_hs[frame_id].append([det_id, x, y, w_hs, h_hs, conf, X, Y, Z])
                det_id += 1
            for hs_idx in unmatched_hs:
                x, y, w, h, conf, X, Y, Z = det_hs[frame_id][hs_idx][1:]
                w_wb = w / width_ratio
                h_wb = h / height_ratio
                det_hs[frame_id][hs_idx][0] = det_id
                det_wb[frame_id].append([det_id, x, y, w_wb, h_wb, conf, X, Y, Z])
                det_id += 1

        for frame_id in diff_wb_frames:
            det_hs[frame_id] = det_wb[frame_id]
            det_id = 1
            for i in range(len(det_hs[frame_id])):
                det_hs[frame_id][i][0] = det_id
                det_wb[frame_id][i][0] = det_id
                det_hs[frame_id][i][3] *= width_ratio
                det_hs[frame_id][i][4] *= height_ratio
                det_id += 1

        for frame_id in diff_hs_frames:
            det_wb[frame_id] = det_hs[frame_id]
            det_id = 1
            for i in range(len(det_wb[frame_id])):
                det_wb[frame_id][i][0] = det_id
                det_hs[frame_id][i][0] = det_id
                det_wb[frame_id][i][3] /= width_ratio
                det_wb[frame_id][i][4] /= height_ratio
                det_id += 1

        save_detection(os.path.join(video_path, video, "detection/gt_wb_comb.txt"), det_wb)
        save_detection(os.path.join(video_path, video, "detection/gt_hs_comb.txt"), det_hs)


if __name__ == "__main__":
    main()
