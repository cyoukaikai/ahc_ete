import os
import pickle

from smrc.object_tracking.data_hub import Query
from smrc.object_tracking.tracker import TrackerSMRC
from smrc.object_tracking.ds import DeepSort

from object_tracking.ahc_ete.metric import PairwiseClusterMetric
from object_tracking.ahc_ete._ahc import HeapCluster
from object_tracking.ahc_ete.ahc_pdist_with_st_const import AHCMultiSequenceTracker


class FalsePositive(Query):
    def __init__(self):
        super().__init__()

        self.false_positive_dets = {}

    def _move_det_to_false_positives(self, false_positives, fp_flag=''):
        """
        Python 3.7.7 (default, Mar 26 2020, 15:48:22)
        [GCC 7.3.0] on linux
        a = {1:{'feature':[2,3]}, 2:{'feature': [4, 5]}}
        a
        Out[3]: {1: {'feature': [2, 3]}, 2: {'feature': [4, 5]}}
        b = {}
        b[1] = a[1].copy()
        b
        Out[6]: {1: {'feature': [2, 3]}}
        del a[1]
        a
        Out[8]: {2: {'feature': [4, 5]}}
        b
        Out[9]: {1: {'feature': [2, 3]}}

        :param false_positives:
        :param fp_flag:
        :return:
        """
        assert isinstance(false_positives, list)
        if len(false_positives) > 0:
            for global_bbox_id in set(false_positives):
                # we must copy the data, otherwise, del will delete the data
                # We do not need deep copy, copy() is enough (refer to the example
                # in the comment section of this function)
                self.false_positive_dets[global_bbox_id] = self.video_detected_bbox_all[global_bbox_id].copy()
                self.false_positive_dets[global_bbox_id]['fp_flag'] = fp_flag

                image_path = self.IMAGE_PATH_LIST[self.get_image_id(global_bbox_id)]
                self.frame_dets[image_path].remove(global_bbox_id)

                del self.video_detected_bbox_all[global_bbox_id]

            assert sum([len(v) for k, v in self.frame_dets.items()]) == len(self.video_detected_bbox_all)

    def count_num_marked(self, fp_flag_):
        count_ = len([k for k, v in self.false_positive_dets.items() if v['fp_flag'] == fp_flag_])
        return count_

    def _restore_from_false_positive(self, fp_flag):
        """Mark the ['fp_flag'] to 'restored'
        Restore the detections in self.false_positive_dets with give fp_flag to self.video_detected_bbox_all.
        Later the restored detections will be used for tracking.
        :param fp_flag: set([value['fp_flag'] for x, value in self.false_positive_dets.items()])
        :return:
        """
        count1, count2 = len(self.false_positive_dets), len(self.video_detected_bbox_all)
        if len(self.false_positive_dets) > 0:
            global_bbox_id_list = self.false_positive_dets.keys()
            for global_bbox_id in global_bbox_id_list:
                # we must copy the data, otherwise, del will delete the data
                if self.false_positive_dets[global_bbox_id]['fp_flag'] == fp_flag:
                    self.video_detected_bbox_all[global_bbox_id] = self.false_positive_dets[global_bbox_id].copy()

                    # self.frame_dets[img_path] = [] for the first initialization
                    image_path = self.IMAGE_PATH_LIST[self.get_image_id(global_bbox_id)]
                    self.frame_dets[image_path] += [global_bbox_id]

                    # mark this detection as 'restored'
                    self.false_positive_dets[global_bbox_id]['fp_flag'] = 'restored'
                    # del self.false_positive_dets[global_bbox_id]
        count1_new, count2_new = len(self.false_positive_dets), len(self.video_detected_bbox_all)
        # print(f'{count1 - count1_new} detections are restored from '
        #       f'self.false_positive_dets')
        print(f'The detections in self.video_detected_bbox_all increased from '
              f'{count2} to {count2_new} [{count2_new - count2}]'
              f' added')
        # assert count1 - count1_new == count2_new - count2

        assert sum([len(v) for k, v in self.frame_dets.items()]) == len(self.video_detected_bbox_all)
        return count2_new - count2  # return number of restored detections

    def _get_false_positive_global_bbox_id_list(self):
        return self.false_positive_dets.keys()

    def _false_positives_by_fp_flag(self, fp_flag):
        global_bbox_id_list = self._get_false_positive_global_bbox_id_list()
        return [x for x in global_bbox_id_list if self.false_positive_dets[x]['fp_flag'] == fp_flag]

    def _get_restored_det_ids(self):
        return self._false_positives_by_fp_flag(fp_flag='restored')


def load_pkl_file(data_file):
    if os.path.isfile(data_file):
        f = open(data_file, 'rb')
        data = pickle.load(f)
        f.close()
        print(f'data loaded from {data_file} .')
        return data
    else:
        return None


def generate_pkl_file(data, pkl_file_name):
    f = open(pkl_file_name, 'wb')
    pickle.dump(data, f)
    f.close()
    print(f'{pkl_file_name} saved ...')


class AHC_ETE(FalsePositive, AHCMultiSequenceTracker, TrackerSMRC,
              PairwiseClusterMetric,
              DeepSort, HeapCluster):
    def __init__(self):
        super(FalsePositive, self).__init__()
        super(AHCMultiSequenceTracker, self).__init__()
        super(TrackerSMRC, self).__init__()
        super(PairwiseClusterMetric, self).__init__()
        super(DeepSort, self).__init__()

        self.Tracker_Params = dict()
        self.false_positive_dets = {}

    def save_ahc_progress(self, data_file_name):
        GlobalKnowledge = {'n_samples': self.n_samples_, 'used_node': self.used_node_,
                           'clusters': self.clusters_, 'parent': self.parent_,
                           'children': self.children_, 'distances': self.distances_,
                           'false_positives': self.false_positive_dets}
        generate_pkl_file(data=GlobalKnowledge, pkl_file_name=data_file_name)

    def save_expert_progress(self, seq_dir, exp_config):
        dir_name, sequence_name = os.path.dirname(seq_dir).replace('train', ''), \
                                  os.path.basename(seq_dir)
        result_file_name = os.path.join(dir_name, exp_config['expert_name'], sequence_name)
        clusters = self.heap_clusters_to_clusters(self.clusters_)
        clusters = self.sorted_clusters_based_on_length(clusters)
        self.clusters = self.delete_cluster_with_length_thd(clusters=clusters, track_min_length_thd=3)

        to_save_dir_name = os.path.dirname(result_file_name)
        if not os.path.exists(to_save_dir_name):
            os.makedirs(to_save_dir_name)
        self.save_tracking_result_mot_format(
            result_file_name
        )
        # data_file_name = os.path.join(dir_name, exp_config['expert_name'], sequence_name + '.pkl')
        # self.save_ahc_progress(data_file_name)
        # # load_pkl_file(data_file_name)

    def offline_tracking(
            self, video_detection_list, video_feature_list=None,
            **kwargs
    ):
        self.init_tracking_tool(
            video_detection_list, video_feature_list=video_feature_list, **kwargs
        )
        if len(self.video_detected_bbox_all) == 0:
            print('Tracking impossible, no detection loaded, return ...')
            self.clusters = []
            return self.clusters, self.video_detected_bbox_all

        if 'expert_team_config' in self.Tracker_Params:
            if 'preprocessing' in self.Tracker_Params['expert_team_config']:
                exp_config = self.Tracker_Params['expert_team_config']['preprocessing']
                self.remove_ambiguous_detections(exp_config)

            if 'exp_appearance_with_st_const' in self.Tracker_Params['expert_team_config']:
                exp_config_list = self.Tracker_Params['expert_team_config']['exp_appearance_with_st_const']
                for e_k, exp_config in enumerate(exp_config_list):
                    linkage, with_st_const = self._parse_expert_linkage_config(exp_config['linkage'])
                    self.appearance_with_st_const_tracking(
                        linkage=linkage, max_dist_thd=exp_config['max_dist'], exp_config=exp_config
                    )
                    self.save_expert_progress(
                        seq_dir=self.Tracker_Params['visualization_sequence_dir'],
                        exp_config=exp_config
                    )

            if 'generate_tracklet' in self.Tracker_Params['expert_team_config']:
                exp_config_list = self.Tracker_Params['expert_team_config']['generate_tracklet']
                for e_k, exp_config in enumerate(exp_config_list):
                    self.generate_tracklet(exp_config)
                    self.save_expert_progress(
                        seq_dir=self.Tracker_Params['visualization_sequence_dir'],
                        exp_config=exp_config
                    )

            if 'claim_from_false_positive' in self.Tracker_Params['expert_team_config']:
                exp_config_list = self.Tracker_Params['expert_team_config']['claim_from_false_positive']
                num_cluster_before = len(self.clusters_)

                fp_flags = ['ambiguous_as_fp']
                ambiguous_levels = [0, 1]
                for level_k in ambiguous_levels:
                    fp_flags.append(f'ambiguous_level_{level_k}')

                for e_k, exp_config in enumerate(exp_config_list):
                    if e_k > 0:
                        self.claim_from_false_positive(exp_config)
                        print(f'| progress {num_cluster_before - len(self.clusters_)} detections restored ...')
                        self.save_expert_progress(
                            seq_dir=self.Tracker_Params['visualization_sequence_dir'],
                            exp_config=exp_config
                        )
                    else:
                        # one level by one level
                        for k_, fp_flag in enumerate(fp_flags):
                            # count = self.count_num_marked(fp_flag_=fp_flag)
                            # print(f'Number of detections marked as {fp_flag}: {count}')
                            if e_k == 0:
                                num_restored_det = self._restore_from_false_positive(fp_flag=fp_flag)
                                # if no any detection is restored, then we do not need to proceed for this function
                                print(f'Total number of restored detections for clustering is {num_restored_det}.')
                                if num_restored_det == 0:
                                    continue

                            self.claim_from_false_positive(exp_config)

                            print(f'| progress {num_cluster_before - len(self.clusters_)} detections restored ...')
                            self.save_expert_progress(
                                seq_dir=self.Tracker_Params['visualization_sequence_dir'],
                                exp_config=exp_config
                            )

        self.clusters = self.heap_clusters_to_clusters(self.clusters_)
        return self.clusters, self.video_detected_bbox_all

    def _computer_pdist(self, global_bbox_id_list, with_st_const):
        self.get_feature_dict_ready()
        return self.computer_pdist(
            global_bbox_id_list,
            metric=self.estimate_appearance_cosine_pdist,
            with_st_const=with_st_const
        )

    def init_tracking_tool(self, video_detection_list, video_feature_list=None, **kwargs):
        # Transfer detections to the object_tracking data format.
        # The detection data and image list are re-initialized in load_detected_bbox_all
        self.load_video_detection_all(
            video_detection_list=video_detection_list
        )
        if video_feature_list is not None:
            self.load_video_feature_list(video_feature_list)
            self.deep_sort_feature_available = True

        self.init_ahc_private_variable()

        # update other settings
        self.init_tracking_params(**kwargs)

    def remove_ambiguous_detections(self, exp_config):
        # conduct preprocessing to filter out ambiguous detections based on
        # the settings of min_detection_score, nms_thd

        def parse_preprocessing_config(preprocessing_config_):
            min_detection_score_, nms_thd_ = None, None
            if 'min_detection_score' in preprocessing_config_:
                min_detection_score_ = preprocessing_config_['min_detection_score']
            if 'nms_thd' in preprocessing_config_:
                nms_thd_ = preprocessing_config_['nms_thd']
            return min_detection_score_, nms_thd_

        temporary_removal_list = []
        if "temporary_removal" in exp_config:
            configs = exp_config["temporary_removal"]
            if isinstance(configs, list) and len(configs) > 0:
                # later we will recover the ambiguous_detections from
                # ambiguous_level_0, ambiguous_level_1, to the last level.
                for level_k, prepro_conf_ in enumerate(configs):
                    min_detection_score, nms_thd = parse_preprocessing_config(
                        prepro_conf_
                    )
                    if min_detection_score is not None or nms_thd is not None:
                        false_positives = self.preprocessing(
                            min_detection_score=min_detection_score,
                            nms_thd=nms_thd
                        )
                        temporary_removal_list.append(false_positives)

        min_detection_score, nms_thd = parse_preprocessing_config(
            exp_config
        )
        if min_detection_score is not None or nms_thd is not None:
            false_positives = self.preprocessing(
                min_detection_score=min_detection_score,
                nms_thd=nms_thd
            )
            fp_flag = 'ambiguous_as_fp'
            # the detections marked as ambiguous_as_fp will not be used in tracking in the first
            # round
            self._move_det_to_false_positives(false_positives, fp_flag=fp_flag)
            count = self.count_num_marked(fp_flag_=fp_flag)
            print(f'Number of detections below to {fp_flag}: {count}')

        # later we will recover the ambiguous_detections from
        # ambiguous_level_0, ambiguous_level_1, to the last level.
        for level_k, false_positives in enumerate(temporary_removal_list):
            fp_flag = f'ambiguous_level_{level_k}'
            not_removed = [x for x in false_positives if x not in self.false_positive_dets]
            self._move_det_to_false_positives(not_removed, fp_flag=fp_flag)
            for x in false_positives:
                self.false_positive_dets[x]['fp_flag'] = fp_flag
            count = self.count_num_marked(fp_flag_=fp_flag)
            print(f'Number of detections below level {fp_flag}: {count}')

        print(f'=============================================')
        fp_flag = 'ambiguous_as_fp'
        count = self.count_num_marked(fp_flag_='ambiguous_as_fp')
        print(f'Number of detections marked as {fp_flag}: {count}')

        for level_k, false_positives in enumerate(temporary_removal_list):
            fp_flag = f'ambiguous_level_{level_k}'
            count = self.count_num_marked(fp_flag_=fp_flag)
            print(f'Number of detections marked as {fp_flag}: {count}')

        count = len([k for k, v in self.false_positive_dets.items() if 'fp_flag' in v])
        print(f'Total number of detections marked: {count}')

    @staticmethod
    def _parse_expert_linkage_config(expert_linkage_config):
        """Parsing the config of a expert, if with string 'tracking', then with spatial-temporal constraints,
        otherwise, without spatial-temporal constraints, e.g.,
            if 'tracking_single', then
                linkage, with_st_const = 'single', True
            if 'single', then
                linkage, with_st_const = 'single', False
        :param expert_linkage_config:
        :return:
        """

        if expert_linkage_config.find('tracking') > -1:
            with_st_const = True
        else:
            with_st_const = False
        linkage = expert_linkage_config.replace('tracking_', '')
        return linkage, with_st_const

    @staticmethod
    def _parse_sequence_config(exp_config):
        subsequence_config = None
        if 'subsequence_config' in exp_config and len(exp_config['subsequence_config']) > 0:
            subsequence_config = exp_config['subsequence_config']
        return subsequence_config

    @staticmethod
    def _parse_filter_config(exp_config):
        filter_config = None
        if 'filter' in exp_config and len(exp_config['filter']) > 0:
            filter_config = exp_config['filter']
        return filter_config

    def _is_valid_merging(self, heap_cluster1, heap_cluster2, **kwargs):
        """Check if two merging are valid based on additional constraints, return true in default
        (i.e., no further checking). Any customized constraints (kf dist, etc.) should be implemented in subclasses.
        :param heap_cluster1:
        :param heap_cluster2:
        :return:
        """
        # return True
        valid = True

        if 'filter_config' in kwargs and kwargs['filter_config'] is not None:
            filter_config = kwargs['filter_config']
            newly_formed_cluster = heap_cluster1 + heap_cluster2
            # heapify(newly_formed_cluster)
            assert len(newly_formed_cluster) >= 2

            # filtering the merging based on the Kalman Filter distance.
            if 'kf' in filter_config:
                dist_kf = self.kf_dist(
                    heap_cluster=newly_formed_cluster,
                    skip_empty_detection=filter_config['kf']['skip_empty_detection'],
                    linkage=filter_config['kf']['linkage']  # complete average
                )
                if dist_kf > filter_config['kf']['max_dist']:
                    valid = False
                    print(f"dist_kf = {dist_kf}, thd: {'%.2f' % filter_config['kf']['max_dist']}")

            if valid and 'appearance' in filter_config:
                appearance_dist = self.appearance_dist_for_two_heap_clusters(
                    heap_cluster1=heap_cluster1,
                    heap_cluster2=heap_cluster2, linkage=filter_config['appearance']['linkage']
                )
                if appearance_dist > filter_config['appearance']['max_dist']:
                    valid = False
                    print(f"appearance_dist = {appearance_dist} > max_dist "
                          f"{'%.2f' % filter_config['appearance']['max_dist']} ....")

            # if valid and 'length' in filter_config:
            #     if not (filter_config['length'][0] <= len(heap_cluster1) <= filter_config['length'][1] and
            #        filter_config['length'][0] <= len(heap_cluster2) <= filter_config['length'][1]):
            #         valid = False

            if valid and 'temp' in filter_config:
                temp_dist = self.temporal_distance(heap_cluster1, heap_cluster2)
                if temp_dist < filter_config['temp'][0] or \
                        temp_dist > filter_config['temp'][1]:
                    valid = False
                    print(f'temp_dist = {temp_dist} not in '
                          f'{filter_config["temp"]} ')

        return valid

    def appearance_with_st_const_tracking(self, linkage, max_dist_thd, exp_config):
        D = self._divide_sequence(subsequence_config=self._parse_sequence_config(exp_config))

        for seq_id in range(len(D) - 1):  # len(D) - 1  range(2)
            global_bbox_id_list = D[seq_id] + D[seq_id + 1]
            # Skip sub sequence tracking if there is less than two detections
            # since no any merging is likely occur.
            if len(global_bbox_id_list) < 2:
                continue

            self.tracking_one_subsequent(
                distance_threshold=max_dist_thd,
                linkage=linkage,
                global_bbox_id_list=global_bbox_id_list
            )

    def generate_tracklet(self, exp_config):
        """
        :param exp_config:
        :return:
        """
        if 'appearance_distance' in exp_config and \
            'linkage' in exp_config['appearance_distance'] and \
                'max_dist' in exp_config['appearance_distance']:

            expert_linkage_config = exp_config['appearance_distance']['linkage']
            max_dist_thd = exp_config['appearance_distance']['max_dist']

            linkage, with_st_const = self._parse_expert_linkage_config(expert_linkage_config)

            filter_config = self._parse_filter_config(exp_config=exp_config)

            print(f'generate_tracklet config: {exp_config}')
            print(f'Total number of detections for clustering is {len(self.video_detected_bbox_all)}.')

            D = self._divide_sequence(subsequence_config=self._parse_sequence_config(exp_config))

            for seq_id in range(len(D) - 1):
                global_bbox_id_list = D[seq_id] + D[seq_id + 1]

                # Skip sub sequence tracking if there is less than two detections
                # since no any merging is likely occur.
                if len(global_bbox_id_list) < 2:
                    continue

                self.tracking_one_subsequent(
                    distance_threshold=max_dist_thd,
                    linkage=linkage,
                    global_bbox_id_list=global_bbox_id_list,
                    major_dist_func=None,
                    filter_config=filter_config
                )

    def claim_from_false_positive(self, exp_config):
        if 'appearance_distance' in exp_config and \
            'linkage' in exp_config['appearance_distance'] and \
                'max_dist' in exp_config['appearance_distance']:

            expert_linkage_config = exp_config['appearance_distance']['linkage']
            max_dist_thd = exp_config['appearance_distance']['max_dist']

            linkage, with_st_const = self._parse_expert_linkage_config(expert_linkage_config)

            filter_config = self._parse_filter_config(exp_config=exp_config)

            D = self._divide_sequence(subsequence_config=self._parse_sequence_config(exp_config))

            num_frame_per_sub_seq = self.estimate_num_frame_per_sub_seq(num_sub_seq=len(D))

            seq_max_length = num_frame_per_sub_seq * 2
            for seq_id in range(len(D) - 1):
                print(f'Processing seq_id {seq_id}/{len(D)} ... ')
                num_cluster_before = len(self.clusters_)

                global_bbox_id_list = D[seq_id] + D[seq_id + 1]

                all_false_positives_to_restore = set(self._get_restored_det_ids())
                restored_det_global_bbox_id_list = list(
                    all_false_positives_to_restore.intersection(set(global_bbox_id_list))
                )

                # Skip sub sequence tracking if there is less than two detections
                # since no any merging is likely occur.
                if len(global_bbox_id_list) < 1 or len(restored_det_global_bbox_id_list) < 1:
                    continue

                # exclude tracks without hole in it, since no merging is likely to occur for them
                param_seq_max_length = seq_max_length if seq_max_length <= 32 else None
                self.filling_hole_one_subsequent(
                    distance_threshold=max_dist_thd,
                    linkage=linkage,
                    global_bbox_id_list=global_bbox_id_list,
                    restored_det_global_bbox_id_list=restored_det_global_bbox_id_list,
                    major_dist_func=None,
                    seq_max_length=param_seq_max_length,
                    filter_config=filter_config
                )

                # change the 'fp_flag' of for detections in self.false_positive_dets if they are
                # linked to existing clusters, so that they can be used as true positives in future
                for x in restored_det_global_bbox_id_list:
                    if len(self.clusters_[self._head_from_global_bbox_id(x)]) > 1:
                        self.false_positive_dets[x]['fp_flag'] = 'restore_succeed'
                        del self.false_positive_dets[x]
                print(f'| progress {num_cluster_before - len(self.clusters_)} detections restored ...')

    def _post_process_for_claiming_restored_det(self):
        """
        Once the dets are moved from self.false_positives to self.video_detected_bbox_all,
        they will stay there for later restoring.
        Here we remove the successfully restored ones from self.false_positives, as they
        are already copy to self.video_detected_bbox_all for restoring, we only need
        to del the keys from self.false_positives.
        :return:
        """

        # ------------------------------------------------------------
        check_list = [k for k, v in self.false_positive_dets.items()
                      if v['fp_flag'] == 'restored']
        linked = [False] * len(check_list)
        for k, global_bbox_id in enumerate(check_list):
            # checking if this detection has been connected to the existing clusters
            if len(self.clusters_[self._head_from_global_bbox_id(global_bbox_id)]) > 1:
                linked[k] = True
        restore_failed_false_positives = [check_list[k] for k in range(len(linked))
                                          if not linked[k]]

        restore_succeed_false_positives = [check_list[k] for k in range(len(linked))
                                           if linked[k]]
        # remove the keys of restored dets from self.false_positive_dets
        for x in restore_succeed_false_positives:
            del self.false_positive_dets[x]

        print(f'{len(restore_succeed_false_positives)} det restored '
              f'{len(restore_failed_false_positives)} failed to be restored ...')
        # ------------------------------------------------------------


