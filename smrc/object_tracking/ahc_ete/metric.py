import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from object_tracking.ahc_ete._ahc import HeapCluster
from object_tracking.ahc_ete import AhcMetric
from object_tracking.deep_sort.kalman_filter import KalmanFilter
from object_tracking.data_hub import Query

import smrc.utils


class TemporalDist(Query):
    @staticmethod
    def is_unique_list(my_list):
        return len(my_list) == len(set(my_list))

    @staticmethod
    def temporal_distance(heap_cluster1, heap_cluster2, allow_negative=False):
        # key, image_id; value, global_bbox_id for a heap_cluster
        image_ids_1 = [x[0] for x in heap_cluster1]
        image_ids_2 = [x[0] for x in heap_cluster2]

        num_overlap = len(image_ids_1) + len(image_ids_2) - len(set(image_ids_1 + image_ids_2))
        if num_overlap > 0:
            if allow_negative:
                return -num_overlap
            else:
                return float('inf')
        else:
            if max(image_ids_1) < image_ids_2[0]:
                return image_ids_2[0] - max(image_ids_1)
            elif max(image_ids_2) < image_ids_1[0]:
                return image_ids_1[0] - max(image_ids_2)
            else:
                return 0


class AppearanceDist(Query):
    def appearance_dist_for_two_heap_clusters(self, heap_cluster1, heap_cluster2, linkage):
        return self.appearance_dist_for_two_clusters(
            cluster1=HeapCluster.heap_cluster_global_bbox_id_list(heap_cluster1),
            cluster2=HeapCluster.heap_cluster_global_bbox_id_list(heap_cluster2),
            linkage=linkage
        )

    def appearance_dist_for_two_clusters(self, cluster1, cluster2, linkage):
        dist_matrix = self.appearance_dist_matrix_for_two_clusters(
            cluster1, cluster2
        )
        return AhcMetric.cluster_distance_by_linkage(
            dist_matrix_or_list=dist_matrix,
            linkage=linkage
        )

    def appearance_dist_matrix_for_two_clusters(self, cluster1, cluster2):
        features_cluster1 = np.array([self.get_feature(x) for x in cluster1])
        features_cluster2 = np.array([self.get_feature(x) for x in cluster2])
        dist_matrix = 1 - cosine_similarity(features_cluster1, features_cluster2)
        return dist_matrix


# modified from ..deep_sort.track.Track
class KFTrack:
    """A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.

    """

    def __init__(self, mean, covariance, init_global_bbox_id=None):
        self.mean = mean
        self.covariance = covariance

        self.global_bbox_id_list = []
        if init_global_bbox_id is not None:
            self.global_bbox_id_list = [init_global_bbox_id]

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)

    def update(self, kf, detection_bbox_xyah):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection_bbox_xyah : the new observed detection
        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection_bbox_xyah)

    def gating_distance_between_kf_and_detections(self, kf, detections_bbox_xyah, only_position=False):
        """
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        :param kf:
        :param detections_bbox_xyah:
        :param only_position:
        :return:
        """
        measurements = np.asarray(detections_bbox_xyah).reshape(-1, 4)
        distance = kf.gating_distance(
            self.mean, self.covariance,
            measurements=measurements,
            only_position=only_position
        )
        return distance

    def mahalanobis_dist(self, kf, detection_bbox_xyah, only_position=False):
        """Return the gating distance between kf and a single detection.

        :param kf:
        :param detection_bbox_xyah:
        :param only_position:
        :return:
        """
        distance = self.gating_distance_between_kf_and_detections(
            kf, detection_bbox_xyah, only_position=only_position
        ).item()
        return distance


class KFDist(Query):

    def kf_dist(
            self, heap_cluster, skip_empty_detection=True, linkage='complete'
    ):
        cluster = HeapCluster.heap_cluster_global_bbox_id_list(heap_cluster)
        return self._kf_distance(
            cluster, skip_empty_detection=skip_empty_detection, linkage=linkage
        )

    # def kf_dist_left_right(
    #         self, heap_cluster1, heap_cluster2, skip_empty_detection=True, linkage='single'
    # ):
    #     min_image_id1, min_image_id2 = heap_cluster1[0][0], heap_cluster2[0][0]
    #     cluster1, cluster2 = [x[1] for x in heap_cluster1], [x[1] for x in heap_cluster2]
    #     # image ids, left [29, 30, 31], right [29], then [29] should not be the right cluster as the gating_dist_list
    #     # return [].
    #     if min_image_id1 < min_image_id2 or \
    #             (min_image_id1 == min_image_id2 and len(cluster1) < len(cluster2)):
    #         return self.kalman_filter_tracker.kf_distance_left_right(
    #             cluster_left=cluster1, cluster_right=cluster2,
    #             skip_empty_detection=skip_empty_detection, linkage=linkage
    #         )
    #     else:  # min_image_id1 > min_image_id2:
    #         return self.kalman_filter_tracker.kf_distance_left_right(
    #             cluster_left=cluster2, cluster_right=cluster1,
    #             skip_empty_detection=skip_empty_detection, linkage=linkage
    #         )

    def _kf_distance(self, cluster, skip_empty_detection=True, linkage='complete'):
        """
        :param cluster:
        :param skip_empty_detection: if True, we do not update kf state once no observation comes, then
        the distance of the state of kf with future new detection is very likely to be large, as no state
        update is conducted. So 'True' is a strong condition for object_tracking, i.e.,
         no allowed sharp change in location.
         In general, we should not skip empty detection for kf in the general sense.
        :param linkage:
        :return:
        """

        cluster = self.sort_cluster_based_on_image_id(cluster)
        image_id_list = self.get_image_id_list_for_cluster(cluster)

        kf, kf_track = self.init_kf_track_with_one_bbox(global_bbox_id=cluster[0])

        gating_distance_list = []
        # for image_id in range(min(image_id_list)+1, max(image_id_list)+1):
        for i in range(1, len(cluster)):
            times_update = image_id_list[i] - image_id_list[i-1]

            kf_track.predict(kf)
            if times_update > 1 and not skip_empty_detection:
                for j in range(1, times_update):
                    kf_track.predict(kf)

            bbox = self.get_single_bbox(cluster[i])
            detection_bbox_xyah = smrc.utils.bbox_to_xyah(bbox, with_class_index=False)

            distance = kf_track.mahalanobis_dist(
                kf=kf, detection_bbox_xyah=detection_bbox_xyah
            )
            # print(f'kf fitting the {ind}th bbox on the cluster, distance = {distance} ...')
            gating_distance_list.append(distance)

            # update the kalman filter only if new observations arrive
            kf_track.update(kf, detection_bbox_xyah)

        return AhcMetric.cluster_distance_by_linkage(
            dist_matrix_or_list=gating_distance_list,
            linkage=linkage
        )

    def init_kf_track_with_one_bbox(self, global_bbox_id):
        kf = KalmanFilter()
        mean, covariance = kf.initiate(
            smrc.utils.bbox_to_xyah(self.get_single_bbox(global_bbox_id), with_class_index=False)
        )

        # initialize kf with the first bbox in cluster_left
        kf_track = KFTrack(
            mean=mean, covariance=covariance, init_global_bbox_id=global_bbox_id
        )
        return kf, kf_track


class PairwiseClusterMetric(KFDist, TemporalDist, AppearanceDist):
    def __init__(self):
        super().__init__()

