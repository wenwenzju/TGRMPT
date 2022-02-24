# summary whole_body and head_shoulder deep sort tracking results
import os
from tabulate import tabulate
from termcolor import colored
import glob


def load_tracking_summary(summary_file):
    assert os.path.exists(summary_file), "Summary file {} doesn't exist.".format(summary_file)
    with open(summary_file, 'r') as f:
        header = f.readline()
        header = header.split()
        values = f.readline()
        values = [float(v) if '.' in v else int(v) for v in values.split()]

        return header, values


metrics = ['HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe', 'AssPr', 'MOTA', 'MOTP', 'IDSW', 'IDF1']
benchmarks = ['iros2022-fisheye-similar-test', 'iros2022-fisheye-tradition-test']
trackers_folder = '../eval/data/trackers/zjlab'


def summaries(tracker_list):
    # extract max_cosine_distance and max_age
    results = {}
    for tl in tracker_list:
        tracker_name = tl.strip(os.sep).split(os.sep)[-1]
        results[tracker_name] = load_tracking_summary(os.path.join(tl, "pedestrian_summary.txt"))

    ret = {}
    header = ["method"] + metrics
    values = []
    for tracker_name in results:
        row = [tracker_name] + [results[tracker_name][1][results[tracker_name][0].index(metric)] for metric in metrics]
        values.append(row)
    return colored(tabulate(values, tablefmt='pipe', floatfmt='0.2f', headers=header, numalign='left'), "cyan")


if __name__ == '__main__':
    for benchmark in benchmarks:
        # summary whole body + head shoulder
        # trackers = glob.glob(os.path.join(trackers_folder, benchmark, "deep_sort_syn_distance0.85*"))
        trackers = [os.path.join(trackers_folder, benchmark, t) for t in ["CenterTrack_wb", "ByteTrack_wb", "deep_sort_concat_distance0.85_budget50_age-1"]]
        res = summaries(trackers)
        print("SOTA", benchmark)
        print(res)
