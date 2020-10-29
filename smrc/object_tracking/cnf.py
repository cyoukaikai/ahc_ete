from tqdm import tqdm

from .data_hub import DataHub
from object_tracking.tracker import TrackerSMRC
from object_tracking._match import *
from object_tracking._connectivity import Connectivity
import copy

######################################
# ConnectNeighbouringFrames
###################################

Epsilon = 1e-5
MaxFrameGap = 3

_MATCH_METRICS = dict(
    bidirectional=BidirectionalMatch,
    hungarian=HungarianMatch,
    ahc=AHCMatch
)


class CNF(DataHub):
    """CNF is short for connect neighbouring frames.
    """
    def __init__(self, match_metric=None):
        super().__init__()

        if match_metric is None:
            self.match_metric = _MATCH_METRICS['bidirectional']
        else:
            self.match_metric = _MATCH_METRICS[match_metric]

        self.frame_dets_ = None
        self.association_params = {}
        self.connectivity = {}
        print(f'Matching metric: {self.match_metric} ...')

    def init_association_params(self, association_params):
        if association_params is not None:
            self.association_params.update(association_params)
            print(f'self.association_params: {self.association_params} ...')

    def _init_connectivity(self, connectivity=None):
        assert len(self.video_detected_bbox_all.keys()) > 0
        if connectivity is None:
            self.connectivity = {
                # right connectedness
                'connectedness': dict.fromkeys(self.video_detected_bbox_all.keys(), None),
                # 'ambiguity': dict.fromkeys(self.video_detected_bbox_all.keys(), None),
                'connectedness_left': dict.fromkeys(self.video_detected_bbox_all.keys(), None),
                'distance': dict.fromkeys(self.video_detected_bbox_all.keys(), None)
            }
        else:
            self.connectivity = connectivity

    def data_association(self, max_dist_thd, max_frame_gap=5, connectivity=None, **kwargs):
        # self.init_association_params(kwargs)

        # initialize the connectedness
        self._init_connectivity(connectivity)

        # connect images in the neighbouring frames
        self.connect_image_sequence(max_dist_thd, max_frame_gap=max_frame_gap)

        # # transfer connectivity to clusters
        clusters = Connectivity.connectedness_to_clusters(
            self.connectivity['connectedness']
        )
        return clusters

    def connect_image_sequence(self, max_dist_thd, max_frame_gap=1):
        """
        # associate detections in neighbouring frames for all frames with given frame gap
        # return connectedness, which will be used to extract clusters.
        :param max_dist_thd:
        :param max_frame_gap:
        :return:
        """
        frame_gap = 1
        while True:
            print(f'Connecting detections with frame gap {frame_gap} ...')
            frame_sequence = self.IMAGE_PATH_LIST[:-frame_gap]
            for image_id, pre_img_path in tqdm(enumerate(frame_sequence)):
                if image_id > 0 and image_id % 100 == 0:
                    print(f'Processing {image_id}/{len(frame_sequence)-1} frame ...')
                # extract only ending bbox of a cluster
                frame_det1 = [
                    k for k in self.frame_dets_[pre_img_path]
                    if self.connectivity['connectedness'][k] is None
                ]

                cur_image_path = self.IMAGE_PATH_LIST[image_id + frame_gap]
                # extract only starting bbox of a cluster
                frame_det2 = [
                    k for k in self.frame_dets_[cur_image_path]
                    if self.connectivity['connectedness_left'][k] is None
                ]

                # print(f'Connecting bounding boxes for frames ({image_id},{image_id + frame_gap})...')
                self.connect_two_frames(
                    frame_det1=frame_det1,
                    frame_det2=frame_det2,
                    max_dist_thd=max_dist_thd
                )

            if frame_gap >= max_frame_gap:
                break
            else:
                self._remove_middle_nodes()
                frame_gap += 1

    # utilities for merging clusters
    def _remove_middle_nodes(self):
        """
        Remove the middle detections in confirmed tracks from self.frame_dets_
        So the future operations only focus on remaining detections
        :return:
        """
        count = 0
        for global_bbox_id, right_node in self.connectivity['connectedness'].items():
            left_node = self.connectivity['connectedness_left'][global_bbox_id]
            if right_node is not None and left_node is not None:
                image_id = self.get_image_id(global_bbox_id)
                # # a point represent a list [image_id, bbox, bbox_id]
                image_path = self.IMAGE_PATH_LIST[image_id]

                # global_bbox_id may have been removed in previous rounds
                # as each time we check middle nodes from all connectivity
                if global_bbox_id in self.frame_dets_[image_path]:
                    self.frame_dets_[image_path].remove(global_bbox_id)
                    # print(f'after remove {global_bbox_id} from {image_path}, image_detection = '
                    #       f'{self.frame_dets_[image_path]}, bbox number = {len(self.frame_dets_[image_path])}')
                    count += 1

        print(f'Total {count} bbox are removed from '
              f'({len(self.video_detected_bbox_all)} bbox) ...')
        num_remain = 0
        for image_path in self.IMAGE_PATH_LIST:
            num_remain += len(self.frame_dets_[image_path])
        print(f'Remaining bbox: {num_remain} ...')
        print(f'====================================')
        return count

    def connect_two_frames(
            self, frame_det1, frame_det2, max_dist_thd
    ):
        if len(frame_det1) == 0 or len(frame_det2) == 0:
            return

        dist_matrix = self.frame_dist_matrix(
            frame_det1, frame_det2
        )
        row_ids, col_ids = self.match_metric(
            dist_matrix, max_distance=max_dist_thd
        )

        for i, j in zip(row_ids, col_ids):
            node_i, node_j = frame_det1[i], frame_det2[j]
            self.connectivity['connectedness'][node_i] = node_j
            # self.connectivity['distance'][node_i] = dist_matrix[i, j]
            self.connectivity['connectedness_left'][node_j] = node_i

    def frame_dist_matrix(self, frame_det1, frame_det2):
        """
        This function should be implemented in child class.
        :rtype: np.ndarray, a distance matrix
        """
        pass

    def _frame_bbox_list(self, frame_det):
        return self.get_bbox_list_for_cluster(frame_det)

    def _frame_color_image(self, frame_det):
        image_id = self.get_image_id(frame_det[0])
        image_path = self.IMAGE_PATH_LIST[image_id]
        tmp_img = cv2.imread(image_path)
        return tmp_img

    def _frame_gray_image(self, frame_det):
        color_image = self._frame_color_image(frame_det)
        return cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # def _frame_bbox_list

    def iou_frame_dist_matrix(self, frame_det1, frame_det2):
        bbox_list1 = self._frame_bbox_list(frame_det1)
        bbox_list2 = self._frame_bbox_list(frame_det2)
        return 1 - IouSimMat(bbox_list1, bbox_list2)

    def l2_frame_dist_matrix(self, frame_det1, frame_det2, ignore_class_label_diff):
        bbox_list1 = self._frame_bbox_list(frame_det1)
        bbox_list2 = self._frame_bbox_list(frame_det2)
        # ignore_class_label_diff = \
        #     self.association_params['ignore_class_label_diff']
        return L2FrameBboxDistMat(
            bbox_list1, bbox_list2,
            ignore_class_label_diff=ignore_class_label_diff
        )

    def opt_flow_frame_dist_matrix(
            self, frame_det1, frame_det2,
            connecting_criteria,
            sampling_method=SamplingMethod.Random,
            num_feature_points=NumFeaturePoint,
            bidirectional=False
    ):
        bbox_list1 = self._frame_bbox_list(frame_det1)
        bbox_list2 = self._frame_bbox_list(frame_det2)
        color_frame1 = self._frame_color_image(frame_det1)
        color_frame2 = self._frame_color_image(frame_det2)
        return OptFlowFrameDistMat(
            color_frame1, bbox_list1,
            color_frame2, bbox_list2,
            sampling_method=sampling_method,
            num_feature_points=num_feature_points,
            connecting_criteria=connecting_criteria,
            bidirectional=bidirectional
        )


class CNFTracker(CNF, TrackerSMRC):  # DataHub,
    def __init__(self, match_metric=None):
        super().__init__(match_metric=match_metric)
        # super(Tracker, self).__init__()

        # initial default tracker params
        # The parameters for each method should be specified separately
        self.Tracker_Params = dict(
            max_dist_thd=1-Epsilon,
            max_frame_gap=MaxFrameGap
        )

    def offline_tracking(self, video_detection_list, **kwargs):
        self._display_tracking_params()

        # load detection
        # initialize the self.video_dets, self.frame_dets, self.IMAGE_PATH_LIST
        self.init_tracking_tool(video_detection_list, **kwargs)

        # copy the frame_dets to a private variable self.frame_dets_ so that all later operations are based on
        # self.frame_dets_. Note that self.frame_dets.copy() does not work, as self.frame_dets
        # is a dictionary
        self.frame_dets_ = copy.deepcopy(self.frame_dets)

        # connect neighbouring frames
        self.clusters = self.data_association(
            max_dist_thd=self.Tracker_Params['max_dist_thd'],
            max_frame_gap=self.Tracker_Params['max_frame_gap']
        )

        # self.clusters = self.postprocessing(clusters)
        # self.report_information_of_resulting_clusters(
        #     clusters=self.clusters
        # )

        return self.clusters, self.video_detected_bbox_all


class IoUTracker(CNFTracker):
    def __init__(self, match_metric=None):
        super().__init__(match_metric=match_metric)
        self.Tracker_Params = dict(
            max_dist_thd=1-Epsilon,  # min_iou = Epsilon
            max_frame_gap=MaxFrameGap
        )

    def frame_dist_matrix(self, frame_det1, frame_det2):
        return self.iou_frame_dist_matrix(frame_det1, frame_det2)


class L2Tracker(CNFTracker):
    def __init__(self, match_metric=None):
        super().__init__(match_metric=match_metric)
        self.Tracker_Params = dict(
            max_dist_thd=100,  # max_l2_dist
            max_frame_gap=MaxFrameGap,
            ignore_class_label_diff=True
        )

    def frame_dist_matrix(self, frame_det1, frame_det2):
        return self.l2_frame_dist_matrix(
            frame_det1, frame_det2,
            ignore_class_label_diff=self.Tracker_Params[
                'ignore_class_label_diff'
            ]
        )


class OptFlowPCTracker(CNFTracker):
    def __init__(self, match_metric=None):
        super().__init__(match_metric=match_metric)
        self.Tracker_Params = dict(
            max_dist_thd=1-Epsilon,  # max_l2_dist
            max_frame_gap=MaxFrameGap,
            sampling_method=SamplingMethod.Random,
            num_feature_points=NumFeaturePoint,
            bidirectional=False
        )

    def frame_dist_matrix(self, frame_det1, frame_det2):
        return self.opt_flow_frame_dist_matrix(
            frame_det1, frame_det2,
            sampling_method=self.Tracker_Params['sampling_method'],
            num_feature_points=self.Tracker_Params['num_feature_points'],
            # not allowed to be modified for connecting_criteria
            connecting_criteria=ConnectingCriteria.PointCount,
            bidirectional=self.Tracker_Params['bidirectional']
        )


class OptFlowIoUTracker(CNFTracker):
    def __init__(self, match_metric=None):
        super().__init__(match_metric=match_metric)
        self.Tracker_Params = dict(
            max_dist_thd=1 - Epsilon,  # max_l2_dist
            max_frame_gap=MaxFrameGap,
            sampling_method=SamplingMethod.Random,
            num_feature_points=NumFeaturePoint,
            bidirectional=False
        )

    def frame_dist_matrix(self, frame_det1, frame_det2):
        return self.opt_flow_frame_dist_matrix(
            frame_det1, frame_det2,
            sampling_method=self.Tracker_Params['sampling_method'],
            num_feature_points=self.Tracker_Params['num_feature_points'],
            # not allowed to be modified for connecting_criteria
            connecting_criteria=ConnectingCriteria.IoU,
            bidirectional=self.Tracker_Params['bidirectional']
        )

