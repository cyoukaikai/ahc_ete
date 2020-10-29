import numpy as np
from .cnf import CNFTracker, Epsilon, MaxFrameGap
from ._match import FeatureFrameCosineDistMat
from .ds import DeepSort


class AppearanceCNFTracker(CNFTracker, DeepSort):
    def __init__(self, match_metric=None):
        super(CNFTracker, self).__init__(
            match_metric=match_metric
        )

        # self.encode_features_for_frame_det is from DeepSort
        super(DeepSort, self).__init__()

        self.Tracker_Params = dict(
            max_dist_thd=1 - Epsilon,  # max_l2_dist
            max_frame_gap=MaxFrameGap,
        )

    def frame_dist_matrix(self, frame_det1, frame_det2):
        return self.deep_sort_feature_cosine_dist_matrix(
            frame_det1, frame_det2
        )

    def deep_sort_feature_cosine_dist_matrix(self, frame_det1, frame_det2):
        bbox_list1 = self._frame_bbox_list(frame_det1)
        bbox_list2 = self._frame_bbox_list(frame_det2)
        color_frame1 = self._frame_color_image(frame_det1)
        color_frame2 = self._frame_color_image(frame_det2)

        features_list1 = self.encode_features_for_frame_det(color_frame1, bbox_list1)
        features_list2 = self.encode_features_for_frame_det(color_frame2, bbox_list2)

        return FeatureFrameCosineDistMat(
            np.array(features_list1), np.array(features_list2)
        )
