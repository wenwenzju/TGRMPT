import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import trackeval  # noqa: E402

plots_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'plots'))
tracker_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'trackers'))

# dataset = os.path.join('kitti', 'kitti_2d_box_train')
# classes = ['cars', 'pedestrian']

# dataset = os.path.join('mot_challenge', 'MOT17-train')
# dataset = os.path.join('zjrm', 'fisheye/five_persons/origin_cloth-test')
# dataset = os.path.join('zjlab', 'iros2022-fisheye-similar-test')
dataset = os.path.join('zjlab', 'iros2022-ablation/similar-cloth')
classes = ['pedestrian']

data_fol = os.path.join(tracker_folder, dataset)
trackers = os.listdir(data_fol)
out_loc = os.path.join(plots_folder, dataset)
for cls in classes:
    trackeval.plotting.plot_compare_trackers(data_fol, trackers, cls, out_loc, settings={'gap_val': 2, 'num_to_plot': 20})
