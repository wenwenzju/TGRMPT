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
        s = tracker_name.find('_distance') + len('_distance')
        e = tracker_name.find('_', s)
        dist = tracker_name[s:e]
        s = tracker_name.find('_age') + len('_age')
        age = tracker_name[s:]
        results[(dist, age)] = load_tracking_summary(os.path.join(tl, "pedestrian_summary.txt"))

    distances = list({res[0] for res in results})
    ages = list({res[1] for res in results})
    distances.sort(key=lambda x: float(x))
    ages.sort(key=lambda x: int(x))
    if ages[0] == '-1':
        ages = ages[1:]
        ages.append('-1')

    ret = {}
    for metric in metrics:
        header = [metric] + distances
        values = []
        for age in ages:
            row = [age] + [results[(dist, age)][1][results[(dist, age)][0].index(metric)] for dist in distances]
            values.append(row)
        ret[metric] = colored(tabulate(values, tablefmt='pipe', floatfmt='0.2f', headers=header, numalign='left'), "cyan")

    return ret


if __name__ == '__main__':
    for benchmark in benchmarks:
        # summary whole body + head shoulder
        trackers = glob.glob(os.path.join(trackers_folder, benchmark, "deep_sort_concat*"))
        res = summaries(trackers)
        for metric in res:
            print("Whole body + head shoulder", benchmark, metric)
            print(res[metric])
