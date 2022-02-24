# vim: expandtab:ts=4:sw=4
import numpy as np


class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : List of array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : List of float
        Detector confidence score.
    feature : List of array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """

    def __init__(self, tlbrs, confidences, features):
        self.tlwh = []
        for tlbr in tlbrs:
            tlbr = np.asarray(tlbr, dtype=np.float)
            tlbr[2:] = tlbr[2:] - tlbr[:2]
            self.tlwh.append(tlbr)
        self.confidence = [float(confidence) for confidence in confidences]
        self.feature = [np.asarray(feature, dtype=np.float32) for feature in features]

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        rets = []
        for tlwh in self.tlwh:
            ret = tlwh.copy()
            ret[2:] += ret[:2]
            rets.append(ret)
        return rets

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        rets = []
        for tlwh in self.tlwh:
            ret = tlwh.copy()
            ret[:2] += ret[2:] / 2
            ret[2] /= ret[3]
            rets.append(ret)
        return rets
