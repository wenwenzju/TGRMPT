# vim: expandtab:ts=4:sw=4
import numpy as np
import colorsys
from tqdm import tqdm
from .image_viewer import ImageViewer


def create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255*r), int(255*g), int(255*b)


class NoVisualization(object):
    """
    A dummy visualization object that loops through all frames in a given
    sequence to update the tracker without performing any visualization.
    """

    def __init__(self, seq_info):
        self.frame_idx = seq_info["min_frame_idx"]
        self.last_idx = seq_info["max_frame_idx"]

    def set_image(self, image):
        pass

    def draw_groundtruth(self, track_ids, boxes):
        pass

    def draw_detections(self, detections):
        pass

    def draw_trackers(self, trackers):
        pass

    def run(self, frame_callback):
        # while self.frame_idx <= self.last_idx:
        for frame_idx in tqdm(range(self.frame_idx, self.last_idx+1)):
            frame_callback(self, frame_idx)
            # self.frame_idx += 1


class Visualization(object):
    """
    This class shows tracking output in an OpenCV image viewe10.00  | 10.88 | 11.02  | 11.22 | 11.28  | 11.18 | 11.19  | 11.13 | 10.95  | 10.85 | 10.71  | 10.68 |
| 100    | 7.63  | 8.18   | 8.91  | 9.70   | 10.62 | 11.46  | 12.92 | 13.25  | 13.39 | 13.63  | 13.57 | 13.28  | 13.32 | 12.76  | 12.29 | 11.81  | 11.53 |
| 500    | 7.66  | 8.22   | 9.16  | 10.16  | 11.62 | 13.10  | 15.79 | 16.62  | 17.89 | 17.78  | 16.94 | 16.12  | 14.74 | 13.08  | 12.44 | 11.40  | 11.19 |
| -1     | 6.72  | 7.07   | 7.71  | 8.14   | 9.01  | 10.72  | 13.09 | 15.40  | 18.69 | 23.86  | 27.68 | 35.14  | 41.37 | 43.85  | 48.30 | 48.93  | 52.39 |
Header shoulder iros2022-fisheye-tradition-test DetRe
| DetRe   | 0.1   | 0.15   | 0.2   | 0.25   | 0.3   | 0.35   | 0.4   | 0.45   | 0.5   | 0.55   | 0.6   | 0.65   | 0.7   | 0.75   | 0.8   | 0.85   | 0.9   |
|:--------|:------|:-------|:------|:-------|:------|:-------|:------|:-------|:------|:-------|:------|:-------|:------|:-------|:------|:-------|:------|
| 30      | 64.85 | 65.22  | 65.56 | 65.78  | 65.94 | 66.08  | 66.16 | 66.24  | 66.29 | 66.32  | 66.35 | 66.36  | 66.38 | 66.39  | 66.40 | 66.42  | 66.43 |
| 100     | 64.88 | 65.26  | 65.63 | 65.92  | 66.12 | 66.32  | 66.44 | 66.51  | 66.59 | 66.67  | 66.69 | 66.74  | 66.73 | 66.72  | 66.77 | 66.80  | 66.83 |
| 500     | 64.88 | 65.27  | 65.67 | 65.99  | 66.25 | 66.46  | 66.65 | 66.75  | 66.83 | 66.92  | 67.03 | 67.09  | 67.18 | 67.20  | 67.13 | 67.27  | 67.32 |
| -1      | 64.62 | 64.78  | 65.00 | 65.27  | 65.52 | 65.84  | 66.11 | 66.34  | 66.58 | 66.80  | 66.94 | 67.05  | 67.11 | 67.14  | 67.15 | 67.17  | 67.06 |
Header shoulder iros2022-fisheye-tradition-test DetPr
| DetPr   | 0.1   | 0.15   | 0.2   | 0.25   | 0.3   | 0.35   | 0.4   | 0.45   | 0.5   | 0.55   | 0.6   | 0.65   | 0.7   | 0.75   | 0.8   | 0.85   | 0.9   |
|:--------|:------|:-------|:------|:-------|:------|:-------|:------|:-------|:------|:-------|:------|:-------|:------|:-------|:------|:-------|:------|
| 30      | 76.74 | 76.63  | 76.50 | 76.42  | 76.34 | 76.30  | 76.24 | 76.20  | 76.18 | 76.17  | 76.17 | 76.15  | 76.15 | 76.13  | 76.13 | 76.11  | 76.10 |
| 100     | 76.72 | 76.60  | 76.45 | 76.35  | 76.25 | 76.17  | 76.07 | 75.98  | 75.93 | 75.94  | 75.91 | 75.87  | 75.82 | 75.76  | 75.75 | 75.73  | 75.71 |
| 500     | 76.71 | 76.59  | 76.44 | 76.31  | 76.23 | 76.13  | 76.04 | 75.93  | 75.86 | 75.80  | 75.80 | 75.72  | 75.72 | 75.62  | 75.46 | 75.52  | 75.56 |
| -1      | 76.83 | 76.74  | 76.65 | 76.48  | 76.22 | 76.02  | 75.71 | 75.57  | 75.51 | 75.47  | 75.43 | 75.47  | 75.50 | 75.44  | 75.45 | 75.43  | 75.28 |
Header shoulder iros2022-fisheye-tradition-test AssRe
| AssRe   | 0.1   | 0.15   | 0.2   | 0.25   | 0.3   | 0.35   | 0.4   | 0.45   | 0.5   | 0.55   | 0.6   | 0.65   | 0.7   | 0.75   | 0.8   | 0.85   | 0.9   |
|:--------|:------|:-------|:------|:-------|:------|:-------|:------|:-------|:------|:-------|:------|:-------|:------|:-------|:------|:-------|:------|
| 30      | 7.55  | 8.07   | 8.62  | 9.10   | 9.56  | 10.20  | 11.09 | 11.25  | 11.46 | 11.55  | 11.45 | 11.46  | 11.40 | 11.25  | 11.16 | 11.03  | 10.99 |
| 100     | 7.73  | 8.31   | 9.07  | 9.89   | 10.86 | 11.72  | 13.20 | 13.53  | 13.70 | 13.97  | 13.96 | 13.72  | 13.78 | 13.32  | 12.92 | 12.46  | 12.26 |
| 500     | 7.76  | 8.35   | 9.32  | 10.36  | 11.88 | 13.43  | 16.22 | 17.14  | 18.69 | 18.77  | 18.16 | 17.68  | 16.51 | 15.51  | 14.65 | 14.07  | 13.70 |
| -1      | 6.79  | 7.15   | 7.82  | 8.26   | 9.15  | 10.92  | 13.38 | 15.82  | 19.29 | 24.82  | 28.97 | 37.50  | 44.63 | 48.24  | 54.67 | 57.19  | 60.86 |
Header shoulder iros2022-fisheye-tradition-test AssPr
| AssPr   | 0.1   | 0.15   | 0.2   | 0.25   | 0.3   | 0.35   | 0.4   | 0.45   | 0.5   | 0.55   | 0.6   | 0.65   | 0.7   | 0.75   | 0.8   | 0.85   | 0.9   |
|:--------|:------|:-------|:------|:-------|:------|:-------|:------|:-------|:------|:-------|:------|:-------|:------|:-------|:------|:-------|:------|
| 30      | 86.75 | 86.16  | 85.73 | 85.20  | 84.73 | 84.27  | 84.43 | 84.05  | 83.95 | 82.86  | 82.30 | 81.87  | 81.42 | 80.26  | 79.55 | 78.73  | 78.49 |
| 100     | 86.74 | 86.11  | 85.62 | 84.91  | 84.15 | 83.77  | 84.33 | 83.48  | 82.29 | 80.78  | 78.57 | 76.07  | 75.43 | 72.64  | 69.27 | 66.64  | 64.54 |
| 500     | 86.74 | 86.10  | 85.55 | 84.71  | 83.87 | 83.10  | 82.97 | 81.69  | 79.24 | 74.73  | 69.93 | 63.38  | 58.03 | 50.11  | 47.03 | 42.58  | 41.59 |
| -1      | 87.08 | 86.97  | 86.76 | 86.53  | 86.24 | 85.79  | 85.19 | 84.60  | 83.93 | 82.93  | 81.86 | 79.88  | 79.39 | 76.07  | 71.43 | 67.73  | 69.22 |
Header shoulder iros2022-fisheye-tradition-test MOTA
| MOTA   | 0.1   | 0.15   | 0.2   | 0.25   | 0.3   | 0.35   | 0.4   | 0.45   | 0.5   | 0.55   | 0.6   | 0.65   | 0.7   | 0.75   | 0.8   | 0.85   | 0.9   |
|:-------|:------|:-------|:------|:-------|:------|:-------|:------|:-------|:------|:-------|:------|:-------|:------|:-------|:------|:-------|:------|
| 30     | 48.30 | 48.66  | 48.97 | 49.17  | 49.27 | 49.39  | 49.44 | 49.49  | 49.53 | 49.55  | 49.55 | 49.55  | 49.56 | 49.55  | 49.55 | 49.55  | 49.54 |
| 100    | 48.30 | 48.68  | 49.00 | 49.21  | 49.32 | 49.48  | 49.52 | 49.59  | 49.63 | 49.66  | 49.62 | 49.59  | 49.58 | 49.54  | 49.50 | 49.46  | 49.43 |
| 500    | 48.29 | 48.67  | 49.00 | 49.20  | 49.31 | 49.47  | 49.53 | 49.66  | 49.66 | 49.68  | 49.62 | 49.54  | 49.53 | 49.43  | 49.38 | 49.34  | 49.30 |
| -1     | 48.11 | 48.16  | 48.32 | 48.39  | 48.39 | 48.47  | 48.41 | 48.52  | 48.77 | 49.03  | 49.13 | 49.48  | 49.67 | 49.75  | 49.85 | 49.84  | 49.78 |
Header shoulder iros2022-fisheye-tradition-test MOTP
| MOTP   | 0.1   | 0.15   | 0.2   | 0.25   | 0.3   | 0.35   | 0.4   | 0.45   | 0.5   | 0.55   | 0.6   | 0.65   | 0.7   | 0.75   | 0.8   | 0.85   | 0.9   |
|:-------|:------|:-------|:------|:-------|:------|:-------|:------|:-------|:------|:-------|:------|:-------|:------|:-------|:------|:-------|:------|
| 30     | 86.62 | 86.59  | 86.57 | 86.56  | 86.55 | 86.54  | 86.54 | 86.53  | 86.53 | 86.53  | 86.53 | 86.53  | 86.52 | 86.52  | 86.52 | 86.52  | 86.52 |
| 100    | 86.62 | 86.60  | 86.58 | 86.57  | 86.57 | 86.56  | 86.56 | 86.55  | 86.54 | 86.54  | 86.55 | 86.54  | 86.54 | 86.54  | 86.53 | 86.53  | 86.53 |
| 500    | 86.62 | 86.60  | 86.59 | 86.59  | 86.60 | 86.61  | 86.60 | 86.60  | 86.61 | 86.61  | 86.62 | 86.62  | 86.62 | 86.61  | 86.61 | 86.59  | 86.60 |
| -1     | 86.62 | 86.62  | 86.61 | 86.59  | 86.58 | 86.58  | 86.55 | 86.53  | 86.54 | 86.50  | 86.48 | 86.47  | 86.46 | 86.44  | 86.42 | 86.40  | 86.38 |
Header shoulder iros2022-fisheye-tradition-test IDSW
| IDSW   | 0.1   | 0.15   | 0.2   | 0.25   | 0.3   | 0.35   | 0.4   | 0.45   | 0.5   | 0.55   | 0.6   | 0.65   | 0.7   | 0.75   | 0.8   | 0.85   | 0.9   |
|:-------|:------|:-------|:------|:-------|:------|:-------|:------|:-------|:------|:-------|:------|:-------|:------|:-------|:------|:-------|:------|
| 30     | 2840  | 2566   | 2320  | 2180   | 2071  | 1971   | 1911  | 1885   | 1851  | 1831   | 1833  | 1837   | 1850  | 1873   | 1890  | 1922   | 1938  |
| 100    | 2860  | 2571   | 2316  | 2176   | 2066  | 1954   | 1901  | 1871   | 1859  | 1850   | 1887  | 1964   | 1982  | 2037   | 2113  | 2174   | 2234  |
| 500    | 2870  | 2579   | 2335  | 2204   | 2108  | 2028   | 1997  | 1985   | 1966  | 1999   | 2066  | 2196   | 2252  | 2422   | 2492  | 2598   | 2637  |
| -1     | 3032  | 2998   | 2897  | 2835   | 2802  | 2696   | 2695  | 2661   | 2487  | 2282   | 2134  | 1861   | 1654  | 1548   | 1413  | 1377   | 1380  |
Header shoulder iros2022-fisheye-tradition-test IDF1
| IDF1   | 0.1   | 0.15   | 0.2   | 0.25   | 0.3   | 0.35   | 0.4   | 0.45   | 0.5   | 0.55   | 0.6   | 0.65   | 0.7   | 0.75   | 0.8   | 0.85   | 0.9   |
|:-------|:------|:-------|:------|:-------|:------|:-------|:------|:-------|:------|:-------|:------|:-------|:------|:-------|:------|:-------|:------|
r.
    """

    def __init__(self, seq_info, update_ms):
        image_shape = seq_info["image_size"][::-1]
        aspect_ratio = float(image_shape[1]) / image_shape[0]
        image_shape = 1024, int(aspect_ratio * 1024)
        self.viewer = ImageViewer(
            update_ms, image_shape, "Figure %s" % seq_info["sequence_name"])
        self.viewer.thickness = 2
        self.frame_idx = seq_info["min_frame_idx"]
        self.last_idx = seq_info["max_frame_idx"]

    def run(self, frame_callback):
        self.viewer.run(lambda: self._update_fun(frame_callback))

    def _update_fun(self, frame_callback):
        if self.frame_idx > self.last_idx:
            return False  # Terminate
        frame_callback(self, self.frame_idx)
        self.frame_idx += 1
        return True

    def set_image(self, image):
        self.viewer.image = image

    def draw_groundtruth(self, track_ids, boxes):
        self.viewer.thickness = 2
        for track_id, box in zip(track_ids, boxes):
            self.viewer.color = create_unique_color_uchar(track_id)
            self.viewer.rectangle(*box.astype(np.int), label=str(track_id))

    def draw_detections(self, detections):
        self.viewer.thickness = 2
        self.viewer.color = 0, 0, 255
        for i, detection in enumerate(detections):
            self.viewer.rectangle(*detection.tlwh[0])

    def draw_trackers(self, tracks):
        self.viewer.thickness = 2
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            self.viewer.color = create_unique_color_uchar(track.track_id)
            self.viewer.rectangle(
                *track.to_tlwh().astype(np.int), label=str(track.track_id))
            # self.viewer.gaussian(track.mean[:2], track.covariance[:2, :2],
            #                      label="%d" % track.track_id)
#
