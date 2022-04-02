# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment

from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

from detection.detection import Detector
from embedding.embedding import Embedding
import time


def iou(bbox, candidates):
    """Computer intersection over candidates.

    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, bottom right x, bottom right y)`.
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
    bbox = np.asarray(bbox)
    candidates = np.asarray(candidates)
    bbox_tl, bbox_br = bbox[:2], bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, 2:]
    candidates_wh = candidates_br - candidates_tl

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_candidates = candidates_wh.prod(axis=1)
    return area_intersection / area_candidates, area_candidates


def create_detections(detectors, extractors, image_names):
    detect_wb, detect_hs = detectors
    extract_wb, extract_hs = extractors

    for img_file in image_names:
        image_o = cv2.imread(img_file)
        image = cv2.cvtColor(image_o, cv2.COLOR_BGR2RGB)
        wb_bbx, wb_conf = detect_wb(image)
        hs_bbx, hs_conf = detect_hs(image)
        detected_bbx = []
        detected_conf = []

        if len(hs_bbx) == 0 and len(wb_bbx) > 0:
            for box, conf in zip(wb_bbx, wb_conf):
                detected_bbx.append([box])
                detected_conf.append([conf])
        elif len(wb_bbx) > 0:
            cost_matrix = np.zeros((len(wb_bbx), len(hs_bbx)))
            for row in range(len(wb_bbx)):
                iou_score, area = iou(wb_bbx[row], hs_bbx)
                iou_score = 1. - iou_score
                idx = np.where(iou_score <= 0.4)[0]
                if len(idx) > 1:
                    # If there are more than one hs detection intersect with wb detection of score no more than 0.4, we select one hs detection with max area
                    to_keep = idx[area[idx].argmax()]
                    for i in idx:
                        if i != to_keep:
                            iou_score[i] = 0.41
                cost_matrix[row, :] = iou_score

            # Handle one hs detection matched to more than one wb detection
            areas = np.array(wb_bbx)[:, 2:].prod(axis=1)
            for col in range(len(hs_bbx)):
                iou_score = cost_matrix[:, col]
                idx = np.where(iou_score <= 0.4)[0]
                if len(idx) > 1:
                    to_keep = idx[areas[idx].argmin()]
                    for i in idx:
                        if i != to_keep:
                            cost_matrix[i, col] = 0.41

            row_indices, col_indices = linear_assignment(cost_matrix)
            matches, unmatched_wb, unmatched_hs = [], [], []
            unmatched_hs = [col for col in range(len(hs_bbx)) if col not in col_indices ]
            unmatched_wb = [row for row in range(len(wb_bbx)) if row not in row_indices]
            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] > 0.4:
                    unmatched_wb.append(row)
                    unmatched_hs.append(col)
                else:
                    matches.append((row, col))
            for match in matches:
                wb_idx, hs_idx = match
                detected_bbx.append([wb_bbx[wb_idx], hs_bbx[hs_idx]])
                detected_conf.append([wb_conf[wb_idx], hs_conf[hs_idx]])
            for wb in unmatched_wb:
                detected_bbx.append([wb_bbx[wb]])
                detected_conf.append([wb_conf[wb]])

        detected_wb = [box[0] for box in detected_bbx]
        detected_hs = [box[1] for box in detected_bbx if len(box) > 1]
        wb_features = extract_wb(image, detected_wb)
        hs_features = extract_hs(image, detected_hs)
        detections = []
        hs_idx = 0
        for i in range(len(detected_bbx)):
            if len(detected_bbx[i]) > 1:
                detections.append(Detection(detected_bbx[i], detected_conf[i], [wb_features[i], hs_features[hs_idx]]))
                hs_idx += 1
            else:
                detections.append(Detection(detected_bbx[i], detected_conf[i], [wb_features[i]]))
        yield detections


def gather_sequence_info(sequence_dir):
    """Gather sequence information, such as image filenames, detections, embedding features

    Parameters
    ----------
    sequence_dir : str
        Path to the image sequence directory.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    files = os.listdir(sequence_dir)
    files.sort()
    image_filenames = {i: os.path.join(sequence_dir, f) for i, f in enumerate(files)}

    image = cv2.imread(next(iter(image_filenames.values())),
                       cv2.IMREAD_GRAYSCALE)
    image_size = image.shape

    min_frame_idx = min(image_filenames.keys())
    max_frame_idx = max(image_filenames.keys())

    seq_info = {
        "sequence_name": sequence_dir,
        "image_filenames": image_filenames,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
    }
    return seq_info


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def run(sequence_dir, detection_weights, embedding_weights, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, max_age, display):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_weights : (whole body detection weight file, head shoulder detection weight file)
        Path to the detection weight file.
    embedding_weights : (whole body embedding weight file, head shoulder embedding weight file)
        Path to the embedding weight file.
    min_confidence : List of float.
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    display : bool
        If True, show visualization of intermediate tracking results.

    """
    seq_info = gather_sequence_info(sequence_dir)
    if not isinstance(min_confidence, (list, tuple)):
        min_confidence = [min_confidence] * 2
    detect_wb = Detector(detection_weights[0], min_confidence[0])
    detect_hs = Detector(detection_weights[1], min_confidence[1])
    extract_wb = Embedding(embedding_weights[0])
    extract_hs = Embedding(embedding_weights[1])
    image_names = list(seq_info["image_filenames"].values())
    image_names.sort()
    detection_generator = create_detections((detect_wb, detect_hs), (extract_wb, extract_hs), image_names)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, max_age=max_age)
    results = []

    def frame_callback(vis, frame_idx):
        s = time.time()
        detections = detection_generator.__next__()
        # Update tracker.
        tracker.predict()
        tracker.update(detections)

        # Update visualization.
        if display:
            image = cv2.imread(
                seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
            vis.set_image(image.copy())
            vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks)

        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

    # Run tracker.
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=5)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    s = time.time()
    visualizer.run(frame_callback)
    dur = time.time() - s
    print("Total time: {}, total frames: {}, fps: {}".format(dur, len(seq_info["image_filenames"]),
                                                             len(seq_info["image_filenames"])/dur))

    # Store results.
    print("Done ", sequence_dir)
    # print("Saving result to ", output_file)
    # if not os.path.exists(os.path.dirname(output_file)):
    #     os.makedirs(os.path.dirname(output_file))
    # f = open(output_file, 'w')
    # for row in results:
    #     print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
    #         row[0], row[1], row[2], row[3], row[4], row[5]),file=f)


def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--sequence_dir", help="Path to images sequence directory",
        default=None, required=True)
    parser.add_argument(
        "--wb_detection", help="Path to whole body detection weight file",
        default="detection/weights/whole_body.engine",
    )
    parser.add_argument(
        "--hs_detection", help="Path to head shoulder detection weight file",
        default="detection/weights/head_shoulder.engine",
    )
    parser.add_argument(
        "--wb_reid", help="Path to whole body feature extractor weight file",
        default="embedding/weights/whole_body.engine",
    )
    parser.add_argument(
        "--hs_reid", help="Path to head shoulder feature extractor weight file",
        default="embedding/weights/head_shoulder.engine",
    )
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="/tmp/hypotheses.txt")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.", nargs='+',
        default=[0.8, 0], type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=0.9, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", nargs='+', type=float, default=0.85)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=50)
    parser.add_argument(
        "--max_age", help="Maximum number of missed misses before a track is deleted.", type=int, default=-1)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool_string)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.sequence_dir, (args.wb_detection, args.hs_detection), (args.wb_reid, args.hs_reid),
        args.min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, args.nn_budget, args.max_age, args.display)
