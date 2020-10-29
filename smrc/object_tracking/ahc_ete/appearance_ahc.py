###########################################################
# Appearance based ahc (only used for testing the correctness of
# my implementation of AHC)
# We should not inherit this class to make code too complex.
############################################################
import time
from object_tracking.tracker import TrackerSMRC
from object_tracking.ds import DeepSort
from object_tracking.ahc_ete import AHCMultiSequenceTracker, AHCSubSequenceTracker


class AppearanceAHC(AHCMultiSequenceTracker, DeepSort, TrackerSMRC):
    def __init__(self):
        super(AHCMultiSequenceTracker, self).__init__()
        super(DeepSort, self).__init__()
        super(TrackerSMRC, self).__init__()

        self.Tracker_Params = dict(
            max_dist_thd=0.3,  # 1-Epsilon
            linkage='complete'
        )

    def _computer_pdist(self, global_bbox_id_list, wrapper):
        self.get_feature_dict_ready()
        return self.computer_pdist(
            global_bbox_id_list,
            metric=self.estimate_appearance_cosine_pdist,
            wrapper=wrapper
        )

    def offline_tracking_by_ahc_clustering(
            self, video_detection_list,  linkage,  max_dist_thd, **kwargs
    ):
        self.init_tracking_tool(video_detection_list, **kwargs)

        if len(self.video_detected_bbox_all) == 0:
            print('Tracking impossible, no detection loaded, return ...')
            return [], self.video_detected_bbox_all

        self.single_expert_tracking(linkage=linkage, max_dist_thd=max_dist_thd)

        self.clusters = self.heap_clusters_to_clusters(self.clusters_)

        return self.clusters, self.video_detected_bbox_all


###########################################################
# Testing my implementation of ahc against the standard ahc,
# to check consistency and speed
############################################################
class TestAHCSubSequenceTracker(AppearanceAHC, AHCSubSequenceTracker):
    def __init__(self):
        super(AppearanceAHC, self).__init__()

    def test_ahc_with_st(self, video_detection_list, **kwargs):
        """Test my implementation of ahc with spatial-temporal constraint
        against the standard AHC.
        Puring clustering done in 0.22355389595031738 seconds ...
            Cluster 1/26, length: 327
            Cluster 2/26, length: 182
            Cluster 3/26, length: 167
            Cluster 4/26, length: 158
            Cluster 5/26, length: 97
            ...
            Cluster 24/26, length: 7
            Cluster 25/26, length: 5
            Cluster 26/26, length: 2

        Initializing A and inertia done in 1.379732370376587 ...
        1600 th merging, len(inertia) = 40001, len(clusters) = 207, (976, 3210, 0.12) ...
        1700 th merging, len(inertia) = 52077, len(clusters) = 107, (3452, 3343, 0.19) ...
        AHC tracking done in 6.643934488296509 seconds ...
        Puring clustering 29.719609493064066 times faster than AHC tracking
            Cluster 1/26, length: 327
            Cluster 2/26, length: 182
            Cluster 3/26, length: 167
            Cluster 4/26, length: 158

        :param video_detection_list:
        :param kwargs:
        :return:
        """
        # initialize the self.video_dets, self.frame_dets, self.IMAGE_PATH_LIST
        self.init_tracking_tool(video_detection_list, **kwargs)

        if len(self.video_detected_bbox_all) == 0:
            print('Tracking impossible, no detection loaded, return ...')
            return

        self.get_feature_dict_ready()

        # ========================================
        # speed test
        # =======================================
        start_time = time.time()
        check_results = self.ahc_clustering_with_st_const(
            max_distance=self.Tracker_Params['max_dist_thd'],
            linkage=self.Tracker_Params['linkage'],
            to_track_global_bbox_id_list=list(self.video_detected_bbox_all.keys())
        )
        end_time = time.time()
        duration_pure = end_time - start_time
        print("Puring clustering done in %s seconds ..." % (duration_pure,))

        self.report_information_of_resulting_clusters(
            clusters=check_results
        )
        start_time = time.time()

        # # tracking with single linkage test
        # heap_clusters = self.AgglomerativeTracking(
        #     distance_threshold=self.Tracker_Params['max_dist_thd'],
        #     linkage=self.Tracker_Params['linkage'],
        #     global_bbox_id_list=list(self.video_detected_bbox_all.keys())
        # )

        # multiple linkage test

        # the intermediate results are saved in self. ... variables.

        self.AgglomerativeTracking(
            distance_threshold=0.1,
            linkage=self.Tracker_Params['linkage'],
            global_bbox_id_list=list(self.video_detected_bbox_all.keys())
        )
        heap_clusters = \
            self.AgglomerativeTracking(
                distance_threshold=self.Tracker_Params['max_dist_thd'],
                linkage=self.Tracker_Params['linkage'],
                global_bbox_id_list=list(self.video_detected_bbox_all.keys()),
                later_expert_flag=True
            )

        end_time = time.time()
        duration_tracking = end_time - start_time
        print("AHC tracking done in %s seconds ..." % (duration_tracking,))
        print("Puring clustering %s times faster than AHC tracking "
              % (duration_tracking/duration_pure,))
        # the resulting clusters by heap_clusters_to_clusters are
        # already sorted by image id for each cluster
        self.clusters = self.heap_clusters_to_clusters(heap_clusters)

        # self.report_information_of_resulting_clusters(
        #     clusters=self.clusters
        # )
        # =================================================
        # check consistency
        # =================================================
        passed = True
        for cluster in check_results:
            if sorted(cluster) not in self.clusters:
                passed = False
                print(f'Failure detected: cluster diff for two methods: {cluster}')
        assert len(check_results) == len(self.clusters)
        if passed:
            print(f'Consistency test passed')
        else:
            print(f'Consistency Failure')
        return self.clusters, self.video_detected_bbox_all

    def local_ahc_usage_example(
            self, linkage,  max_dist_thd, to_track_global_bbox_id_list, with_st_const,
            later_expert_flag=False):
        """This is the only interface for ahc based tracking with single expert.
        :param linkage:
        :param max_dist_thd:
        :param to_track_global_bbox_id_list:
        :param with_st_const:
        :param later_expert_flag: if later expert, then rebuild the inertia from confirmed clusters
        :return:
        """
        if with_st_const:
            print(f'Tracking with spatial-temporal constraints '
                  f'{linkage} ...')
            # if we use the built-in standard ahc, we need to extract the values
            # of self.cluster_, self.node_, self.distance_, i.e., the intermediate
            # results to continue tracking for later experts.
            heap_clusters = \
                self.AgglomerativeTracking(
                    distance_threshold=max_dist_thd,
                    linkage=linkage,
                    global_bbox_id_list=to_track_global_bbox_id_list,
                    later_expert_flag=later_expert_flag
                )
            clusters = self.heap_clusters_to_clusters(heap_clusters)

            # if linkage in ['complete', 'average']:
            #     clusters = self.ahc_clustering_with_st_const(
            #         max_distance=max_dist_thd,
            #         linkage=linkage,
            #         to_track_global_bbox_id_list=to_track_global_bbox_id_list
            #     )
            # else:  # "single"
            #     heap_clusters = \
            #         self.AgglomerativeTracking(
            #             distance_threshold=max_dist_thd,
            #             linkage=linkage,
            #             global_bbox_id_list=to_track_global_bbox_id_list,
            #             later_expert_flag=later_expert_flag
            #         )
            #     clusters = self.heap_clusters_to_clusters(heap_clusters)
        else:
            print(f'Tracking without spatial-temporal constraints '
                  f'{linkage} ...')
            assert not later_expert_flag, f'Tracking with later_expert_flag = {later_expert_flag} not defined.'

            # average, complete linkage for my implementation are slower than the standard ahc thus
            # we would rather use the standard ahc for those two linkage)
            clusters = self.ahc_clustering_without_constraint(
                max_distance=max_dist_thd,
                linkage=linkage,
                to_track_global_bbox_id_list=to_track_global_bbox_id_list
            )
        return clusters



