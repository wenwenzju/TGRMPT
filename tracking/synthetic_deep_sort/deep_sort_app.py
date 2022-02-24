# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os
import pickle

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker


def gather_sequence_info(sequence_dir, detection_files):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_files : List of str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: List of numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}

    detections = []
    if detection_files is not None and len(detection_files) > 0:
        for detection_file in detection_files:
            assert os.path.exists(detection_file), "Detection file {} doesn't exist.".format(detection_file)
            with open(detection_file, 'rb') as f:
                detections.append(np.array(pickle.load(f)))

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[0][:, 0].min())
        max_frame_idx = int(detections[0][:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    feature_dim = detections[0].shape[1] - 10 if len(detections) > 0 else 0
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info


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
    bbox = np.asarray(bbox)
    candidates = np.asarray(candidates)
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
    return area_intersection / area_candidates, area_candidates


def create_detections(detection_mats, frame_idx, min_height=0, min_confidences=0, image=None):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mats : List of ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.
    min_confidences: List of float, has the same length as detection_mats

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    detection_list = []
    bboxes, confidences, features = [], [], []
    if not isinstance(min_confidences, (list, tuple)):
        min_confidences = [min_confidences] * len(detection_mats)
    for detection_mat, min_confidence in zip(detection_mats, min_confidences):
        frame_indices = detection_mat[:, 0].astype(np.int)
        mask = frame_indices == frame_idx
        bboxes.append([row[2:6] for row in detection_mat[mask] if row[6] >= min_confidence])
        confidences.append([row[6] for row in detection_mat[mask] if row[6] >= min_confidence])
        features.append([row[10:] for row in detection_mat[mask] if row[6] >= min_confidence])
    # for i, boxes in enumerate(bboxes):
    #     for box in boxes:
    #         x1, y1, w, h = np.array(box, dtype=np.int)
    #         x2 = x1 + w
    #         y2 = y1 + h
    #         image = cv2.rectangle(image, (x1, y1), (x2, y2), get_color(i), 2)
    # cv2.imshow("detection2", image)

    # For now, only two kinds of features are used, i.e., whole body and head shoulder
    if len(bboxes) == 2 and len(bboxes[0]) > 0:
        if len(bboxes[1]) == 0:
            for box, conf, feat in zip(bboxes[0], confidences[0], features[0]):
                detection_list.append(Detection(box, conf, [feat]))
        else:
            cost_matrix = np.zeros((len(bboxes[0]), len(bboxes[1])))
            for row in range(len(bboxes[0])):
                iou_score, area = iou(bboxes[0][row], bboxes[1])
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
            areas = np.array(bboxes[0])[:, 2:].prod(axis=1)
            for col in range(len(bboxes[1])):
                iou_score = cost_matrix[:, col]
                idx = np.where(iou_score <= 0.4)[0]
                if len(idx) > 1:
                    to_keep = idx[areas[idx].argmin()]
                    for i in idx:
                        if i != to_keep:
                            cost_matrix[i, col] = 0.41

            row_indices, col_indices = linear_assignment(cost_matrix)
            matches, unmatched_wb, unmatched_hs = [], [], []
            unmatched_hs = [col for col in range(len(bboxes[1])) if col not in col_indices ]
            unmatched_wb = [row for row in range(len(bboxes[0])) if row not in row_indices]
            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] > 0.4:
                    unmatched_wb.append(row)
                    unmatched_hs.append(col)
                else:
                    matches.append((row, col))
            for match in matches:
                wb_idx, hs_idx = match
                detection_list.append(Detection(bboxes[0][wb_idx],
                                                confidences[0][wb_idx],
                                                [features[0][wb_idx], features[1][hs_idx]]))
            for wb in unmatched_wb:
                detection_list.append(Detection(bboxes[0][wb], confidences[0][wb], [features[0][wb]]))

        # ignore unmatched_hs
    elif len(bboxes) > 0:
        for box, conf, feat in zip(bboxes[0], confidences[0], features[0]):
            detection_list.append(Detection(box, conf, [feat]))

    return detection_list


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def run(sequence_dir, detection_file, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, max_age, display):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : List of str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
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
    seq_info = gather_sequence_info(sequence_dir, detection_file)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, max_age=max_age)
    results = []

    def frame_callback(vis, frame_idx):
        # print("Processing frame %05d" % frame_idx)
        if frame_idx == 1950:
            # print(frame_idx)
            pass

        # Load image and generate detections.
        # image = cv2.imread(seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
        detections = create_detections(
            seq_info["detections"], frame_idx, min_detection_height, min_confidence)

        if display:
            image = cv2.imread(seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
            for i, det in enumerate(detections):
                boxes = det.to_tlbr()
                for box in boxes:
                    x1, y1, x2, y2 = np.array(box, dtype=np.int)
                    image = cv2.rectangle(image, (x1, y1), (x2, y2), get_color(i), 2)
            cv2.imshow("detections", image)
            cv2.waitKey(2)

        # Run non-maxima suppression.
        # boxes = np.array([d.tlwh[0] for d in detections])
        # scores = np.array([d.confidence[0] for d in detections])
        # indices = preprocessing.non_max_suppression(
        #     boxes, nms_max_overlap, scores)
        # detections = [detections[i] for i in indices]

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
    visualizer.run(frame_callback)

    # Store results.
    print("Done ", sequence_dir)
    print("Saving result to ", output_file)
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    f = open(output_file, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)


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
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=None, required=True)
    parser.add_argument(
        "--detection_file", help="Path to custom detections.", nargs='+',
        required=True)
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="/tmp/hypotheses.txt")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.", nargs='+',
        default=0.8, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=0.9, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument(
        "--max_age", help="Maximum number of missed misses before a track is deleted.", type=int, default=30)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool_string)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.sequence_dir, args.detection_file, args.output_file,
        args.min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, args.nn_budget, args.max_age, args.display)
