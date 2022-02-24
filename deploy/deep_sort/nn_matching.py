# vim: expandtab:ts=4:sw=4
import numpy as np


def _pdist(a, b):
    """Compute pair-wise squared distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2


def _cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)


def _nn_euclidean_distance(x, y):
    """ Helper function for nearest neighbor distance metric (Euclidean).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest Euclidean distance to a sample in `x`.

    """
    distances = _pdist(x, y)
    return np.maximum(0.0, distances.min(axis=0))


def _nn_cosine_distance(x, y):
    """ Helper function for nearest neighbor distance metric (cosine).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest cosine distance to a sample in `x`.

    """
    distances = _cosine_distance(x, y)
    return distances.min(axis=0)


class TrackLet:
    def __init__(self, nn_budget=None):
        self.nn_budget = nn_budget
        self.wb_features = []
        self.hs_features = []
        self.wb_mask = []
        self.hs_mask = []

    def append(self, features):
        """
        Parameters
        ----------
        :param features: List of features. For now, has 1 or 2 features
        :return:
        """
        if len(features) == 2:
            self.wb_features.append(features[0])
            self.hs_features.append(features[1])
            self.wb_mask.append(1)
            self.hs_mask.append(1)
        else:
            self.wb_features.append(features[0])
            self.hs_features.append([1]*len(features[0]))
            self.wb_mask.append(1)
            self.hs_mask.append(0)
        if self.nn_budget is not None:
            self.wb_features = self.wb_features[-self.nn_budget:]
            self.hs_features = self.hs_features[-self.nn_budget:]
            self.wb_mask = self.wb_mask[-self.nn_budget:]
            self.hs_mask = self.hs_mask[-self.nn_budget:]


class NearestNeighborDistanceMetric(object):
    """
    A nearest neighbor distance metric that, for each target, returns
    the closest distance to any sample that has been observed so far.

    Parameters
    ----------
    metric : str
        Either "euclidean" or "cosine".
    matching_threshold: List of float
        The matching threshold. Samples with larger distance are considered an
        invalid match.
    budget : Optional[int]
        If not None, fix samples per class to at most this number. Removes
        the oldest samples when the budget is reached.

    Attributes
    ----------
    samples : Dict[int -> List[ndarray]]
        A dictionary that maps from target identities to the list of samples
        that have been observed so far.

    """

    def __init__(self, metric, matching_threshold, budget=None):
        self.metric = metric
        if metric == "euclidean":
            self._metric = _pdist
        elif metric == "cosine":
            self._metric = _cosine_distance
        else:
            raise ValueError(
                "Invalid metric; must be either 'euclidean' or 'cosine'")
        if not isinstance(matching_threshold, (tuple, list)):
            self.wb_matching_threshold, self.hs_matching_threshold, self.matching_threshold = [matching_threshold]*3
        else:
            if len(matching_threshold) == 1:
                matching_threshold = matching_threshold * 2
            self.wb_matching_threshold, self.hs_matching_threshold = matching_threshold[:2]
            self.matching_threshold = (self.wb_matching_threshold + self.hs_matching_threshold) / 2
        self.budget = budget
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        """Update the distance metric with new data.

        Parameters
        ----------
        features : List of List of features
                A list of features for every target.
        targets : List
            An integer array of associated target identities.
        active_targets : List[int]
            A list of targets that are currently present in the scene.

        """
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, TrackLet(self.budget)).append(feature)
        self.samples = {k: self.samples[k] for k in active_targets}

    def _distance(self, tracklet, wb_features, wb_mask, hs_features, hs_mask):
        """
        Compute distance between a tracklet and detected features
        :param tracklet: List of TrackLet.
        :param wb_features: List of array. Detected whole body features.
        :param wb_mask: List of array. Indicate whether each wb_feature is available.
        :param hs_features: List of array. Detected head shoulder features.
        :param hs_mask: List of array. Indicate whether each hs_feature is available
        :return: cost matrix
        """
        # whole body cost matrix
        cost_mat1 = self._metric(tracklet.wb_features, wb_features)
        # head shoulder cost matrix
        cost_mat2 = self._metric(tracklet.hs_features, hs_features)

        # whole body mask matrix
        cost_mask1 = np.ones(cost_mat1.shape)
        cost_mask1[np.array(tracklet.wb_mask) == 0, :] = 0
        cost_mask1[:, np.array(wb_mask) == 0] = 0
        # head shoulder mask matrix
        cost_mask2 = np.ones(cost_mat2.shape)
        cost_mask2[np.array(tracklet.hs_mask) == 0, :] = 0
        cost_mask2[:, np.array(hs_mask) == 0] = 0

        # If m1=1 and m2=1, w=0.5, then cost=0.5*cost1+0.5*cost2
        # If m1=1 and m2=0, w=1, then cost=1*cost1
        # if m1=0 and m2=1, w=0, then cost=1*cost2
        # weight_mat = 0.5 * cost_mask1 + 0.5 * (1 - cost_mask2)
        weight_mat = 0.5 * cost_mask1 + 0.5 * (1 - cost_mask2)
        cost_mat = weight_mat * cost_mat1 + (1-weight_mat)*cost_mat2

        # For now, assume whole body is always available
        cost_mat[cost_mask2 == 0] -= (self.wb_matching_threshold - self.matching_threshold)

        if self.metric == 'euclidean':
            return np.maximum(0.0, cost_mat.mean(axis=0))
        else:
            return cost_mat.mean(axis=0)

    def distance(self, features, targets):
        """Compute distance between features and targets.

        Parameters
        ----------
        features : List of List array
        targets : List[int]
            A list of targets to match the given `features` against.

        Returns
        -------
        ndarray
            Returns a cost matrix of shape len(targets), len(features), where
            element (i, j) contains the closest squared distance between
            `targets[i]` and `features[j]`.

        """
        cost_matrix = np.zeros((len(targets), len(features)))
        detectionlet = TrackLet()
        for feat in features:
            detectionlet.append(feat)
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._distance(self.samples[target], detectionlet.wb_features, detectionlet.wb_mask,
                                               detectionlet.hs_features, detectionlet.hs_mask)
        # cost_mat2 = 1. - cost_matrix
        # cost_mat2[cost_mat2 < 0] = 0
        # cost_mat2[cost_mat2 < 1 - self.matching_threshold] = 0
        # cost_mat_denom = cost_mat2.sum(0)[np.newaxis, :] + cost_mat2.sum(1)[:, np.newaxis] - cost_mat2
        # cost_mat2 = cost_mat2 / (cost_mat_denom+np.finfo('float').eps)
        # cost_matrix = 1. - cost_mat2

        return cost_matrix
