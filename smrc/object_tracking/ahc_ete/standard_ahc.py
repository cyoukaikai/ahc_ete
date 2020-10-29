############################
# Standard AHC based tracking
##########################
from itertools import combinations

from object_tracking._metrics import *
from smrc.object_tracking.data_hub import Query

from sklearn.cluster import AgglomerativeClustering


class AHC(Query):
    """Standard AHC algorithm: pdist based clustering without spatial-temporal constraints.
    We can use AHC clustering to conduct pdist (with or without spatial-temporal constraints) based object tracking,
    where the spatial-temporal constraints can be imposed in pdist by setting the corresponding locations
    to be 'inf', here 1e+5 in our implementation.
    However, the tracking is only limit to average and complete linkage, as dist "inf"
    will be taken into account for estimating the dist between a new cluster with existing
    clusters) using the built-in AgglomerativeClustering in sklearn.cluster, and two clusters with
    dist 'inf' (> max_dist) will never be merged in the Standard AHC algorithm.
    The "single" linkage only take the smallest dist of two clusters into account, thus 'inf' of the
    spatial-temporal constraints is simply ignored. That's why we can not use the Standard AHC algorithm
    for conducting pdist based tracking with ["single" linkage + spatial-temporal constraints].

    Note that we need to implement _computer_pdist function to estimate pdist in sub-classes .
    """

    def ahc_clustering_without_constraint(self, max_distance, linkage, to_track_global_bbox_id_list):
        """Standard AHC based clustering with the original defined linkage.
        :param max_distance:
        :param linkage:
        :param to_track_global_bbox_id_list:
        :return:
        """
        clustering_model = self._ahc(
            max_distance, linkage, to_track_global_bbox_id_list,
            with_st_const=False
        )
        clusters = self.labels_to_clusters(clustering_model.labels_, to_track_global_bbox_id_list)
        return clusters

    def ahc_clustering_with_st_const(self, max_distance, linkage, to_track_global_bbox_id_list):
        """AHC based clustering with the spatial-temporal constraints respected.
        The linkages here are modified ones with the constraints coded in.
        It can be used as a naive object tracking algorithm.
        :param max_distance:
        :param linkage:
        :param to_track_global_bbox_id_list:
        :return:
        """
        assert linkage in ['complete', 'average'], \
            'Linkage other than complete, average are not defined.\n ' \
            'Note single linkage does not take the not-in-same-image constraints into consideration, ' \
            'and thus is not available here.'
        clustering_model = self._ahc(
            max_distance, linkage,
            to_track_global_bbox_id_list, with_st_const=True
        )
        clusters = self.labels_to_clusters(clustering_model.labels_, to_track_global_bbox_id_list)
        return clusters

    def _ahc(self, max_distance, linkage, global_bbox_id_list, with_st_const):
        pdist = self._computer_pdist(global_bbox_id_list, with_st_const=with_st_const)
        clustering_model = self._agglomerative_hierarchical_clustering(
            pdist=pdist,
            max_distance=max_distance,
            linkage=linkage
        )
        return clustering_model

    @staticmethod
    def _agglomerative_hierarchical_clustering(pdist, max_distance, linkage):
        """The basic function used to generate a clustering results given the pdist, max_distance,
        and linkage. It has nothing to do with whether it is object tracking or not.

        :param pdist:
        :param max_distance:
        :param linkage:
        :return:
        """
        assert linkage in ['single', 'complete', 'average']

        clustering_model = AgglomerativeClustering(
            n_clusters=None, linkage=linkage,
            distance_threshold=max_distance,
            compute_full_tree=True,
            connectivity=None,
            affinity='precomputed'
        )
        clustering_model.fit(pdist)

        return clustering_model

    @staticmethod
    def labels_to_clusters(labels, global_bbox_id_list):
        """
        The transformation from the labels for pdist to the clusters of the global bbox ids are all done in and only
        in this function.

        labels, np.array()
        refer_tuple = [(i, x) for i, x in enumerate(self.video_feature_dict)]

        # first element in the tuple is the key, the second element is the value
        self.pdist_index_global_bbox_id_reference = dict(refer_tuple)
        :param labels:
        :param global_bbox_id_list:
        :return:
        """
        # extract clusters from clustering result.
        clusters = []
        DetID2GlobalBboxID_Array = np.array([x for x in global_bbox_id_list])
        for i in range(int(np.max(labels) + 1)):
            # we must transfer the ndarry to list, otherwise, the cluster with 1 element will be saved as ndarry type
            # in the final clusters, while cluster with more than 1 element will be saved as list.
            # Different types in the list will cause problem in later processing
            clusters.append(DetID2GlobalBboxID_Array[labels == i].tolist())
        return clusters

    def _computer_pdist(self, global_bbox_id_list, with_st_const):
        """Computing the pairwise distance of the detections and reflect the spatial-temporal constraints
        if wrapper is true.

        :rtype: pdist
        """
        pass

    ###############################################
    # metric definition
    ###############################################

    # 1) pdist estimation
    def computer_pdist(self, global_bbox_id_list, metric, with_st_const=True, **kwargs):
        pdist = metric(global_bbox_id_list, **kwargs)

        # wrapper_pdist_with_same_image_constraint
        if with_st_const:
            mask = self.estimate_same_image_mask(global_bbox_id_list)
            pdist[mask == 0] = BIG_VALUE
        return pdist

    def estimate_same_image_mask(self, global_bbox_id_list):
        """
        slightly slow version, I do not know why it is slow
            ########################################
            image_ids = np.array([[self.get_image_id(x)] for x in global_bbox_id_list])
            # mask(i, j) = 0 if image_ids[i] == image_ids[j], i.e., with same image id
            mask = pairwise_distances(image_ids, metric="hamming")
            ########################################
        tests:
            print(f'Constructing can not link for detections in the same image ... ')
            start_time = time.time()
            mask1 = self.estimate_same_image_mask(global_bbox_id_list)
            print("Initializing mask done in [--- %s seconds ---]" % (time.time() - start_time))

            start_time = time.time()
            ImageIDs = np.array([self.get_image_id(x) for x in global_bbox_id_list])
            n_samples = len(global_bbox_id_list)
            mask = np.ones((n_samples, n_samples), dtype=np.intp)
            min_frame_id, max_frame_id = min(ImageIDs), max(ImageIDs)
            for image_id in range(min_frame_id, max_frame_id + 1):
                # find the pairs of detections in the same image
                inds = np.where(ImageIDs == image_id)[0]  # np.where return a tuple
                if len(inds) > 0:  # with detection in this frame
                    links = list(combinations(inds, 2))  # sort to make i < j
                    for i, j in links:
                        mask[i, j] = mask[j, i] = 0
            np.fill_diagonal(mask, 0)
            print("Initializing mask done in [--- %s seconds ---]" % (time.time() - start_time))

            assert np.array_equal(mask1, mask)
            # >>> True
            # >>> Initializing mask done in [--- 0.02746725082397461 seconds ---]
            # >>> Initializing mask done in [--- 0.01790332794189453 seconds ---]
        :param global_bbox_id_list:
        :return:
        """
        ImageIDs = np.array([self.get_image_id(x) for x in global_bbox_id_list])
        n_samples = len(global_bbox_id_list)
        mask = np.ones((n_samples, n_samples), dtype=np.intp)
        min_frame_id, max_frame_id = min(ImageIDs), max(ImageIDs)
        for image_id in range(min_frame_id, max_frame_id + 1):
            # find the pairs of detections in the same image
            inds = np.where(ImageIDs == image_id)[0]  # np.where return a tuple
            if len(inds) > 0:  # with detection in this frame
                links = list(combinations(inds, 2))  # sort to make i < j
                for i, j in links:
                    mask[i, j] = mask[j, i] = 0
        # We are storing the graph in a list of IntFloatDict
        # Put the diagonal to cannot_link_dist so that we will disable the connection to itself
        np.fill_diagonal(mask, 0)
        return mask

    def estimate_appearance_cosine_pdist(self, global_bbox_id_list, **kwargs):
        # generate feature matrix
        X = np.array([self.get_feature(x) for x in global_bbox_id_list])
        return FeatureCosinePdistMat(X)

    def estimate_iou_pdist(self, global_bbox_id_list, **kwargs):
        bbox_list = self.get_bbox_list_for_cluster(global_bbox_id_list)
        return 1 - IouPairwiseSimMat(bbox_list)

    def estimate_l2_pdist(self, global_bbox_id_list, ignore_class_label_diff, **kwargs):
        bbox_list = self.get_bbox_list_for_cluster(global_bbox_id_list)
        return L2BboxPdistMat(
            bbox_list,
            ignore_class_label_diff=ignore_class_label_diff
        )


