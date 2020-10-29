from tqdm import tqdm
import os
import cv2
from collections import Counter
import numpy as np
import sys

from smrc.utils.det import json_det_to_tracking_format
import smrc.utils
from smrc.utils.annotate.curve_fit import BBoxCurveFitting, CurveFitting


class Query:
    def __init__(self):
        # format for self.video_detected_bbox_all:  [
        #   [image id, bbox, global_bbox_id], ...
        # ], where image id (0-index), the frame id in the image sequences of the video
        # bounding box format, [class_id, x1, y1, x2, y2]
        # global_bbox_id, the row index of a bbox in the bbox_list of the whole video

        # Note that no any row in self.video_detected_bbox_all is allowed to be deleted
        # Otherwise, the saved row index information will be inaccurate.
        # If deleting a bbox is necessary, we can make a separate list to record the
        # indices of the bbox to be deleted.
        self.video_detected_bbox_all = {}
        self.frame_dets = {}  # a point represent a list [image_id, bbox, bbox_id]
        self.clusters = []
        self.bbox_cluster_IDs = {}  # {}
        self.IMAGE_PATH_LIST = []

    def get_image_id(self, global_bbox_id):
        """
        :param global_bbox_id: the bbox_id with respect to the bboxes of whole video
        :return:
        image id of this global_bbox_id
        """
        image_id = self.video_detected_bbox_all[global_bbox_id]['image_id']
        return image_id

    def get_single_bbox(self, global_bbox_id):
        """
        :param global_bbox_id: the bbox_id with respect to the bboxes of whole video
        :return: bounding box information [class_id, x1, y1, x2, y2] of this global_bbox_id
        """
        return self.video_detected_bbox_all[global_bbox_id]['bbox']

    def get_feature(self, global_bbox_id):
        return self.video_detected_bbox_all[global_bbox_id]['feature']

    def get_detection_score(self, global_bbox_id):
        return self.video_detected_bbox_all[global_bbox_id]['score']

    def get_image_id_and_bbox(self, global_bbox_id):
        return [self.get_image_id(global_bbox_id),
                self.get_single_bbox(global_bbox_id)]

    def get_image_id_list_sorted(self, global_bbox_id_list):
        image_id_list = self.get_image_id_list_for_cluster(global_bbox_id_list)
        image_id_list.sort()
        return image_id_list

    def get_unique_image_id_list(self, global_bbox_id_list):
        image_id_list = self.get_image_id_list_for_cluster(global_bbox_id_list)
        return list(set(image_id_list))

    @staticmethod
    def assert_cluster_non_empty(global_bbox_id_list):
        assert len(global_bbox_id_list) > 0, \
            print('There is no bbox_id in the global_bbox_id_list, please check.')

    def get_image_id_list_for_cluster(self, global_bbox_id_list):
        """
        :param global_bbox_id_list: the (global) bbox_id list
        :return:
            image id list
        """
        self.assert_cluster_non_empty(global_bbox_id_list)
        return [self.get_image_id(x) for x in global_bbox_id_list]

    def get_bbox_list_for_cluster(self, global_bbox_id_list):
        """
         :param global_bbox_id_list: the (global) bbox_id list
         :return:
             bbox list for the given global_bbox_id_list
         """
        self.assert_cluster_non_empty(global_bbox_id_list)
        return [self.get_single_bbox(x) for x in global_bbox_id_list]

    def get_score_list_for_cluster(self, global_bbox_id_list):
        self.assert_cluster_non_empty(global_bbox_id_list)
        return [self.get_detection_score(x) for x in global_bbox_id_list]

    def get_class_label_list_for_cluster(self, global_bbox_id_list):
        """
        get the class ids for a single cluster
        :param global_bbox_id_list:
        :return:
        """
        # class_ids = []  # class id
        # for global_bbox_id in global_bbox_id_list:
        #     bbox = self.get_single_bbox(global_bbox_id)
        #     class_idx = bbox[0]
        #     class_ids.append(class_idx)

        self.assert_cluster_non_empty(global_bbox_id_list)
        return [self.get_single_bbox(x)[0] for x in global_bbox_id_list]

    def sort_cluster_based_on_image_id(self, cluster):
        # the cluster should have at least one element and all element must be unique
        assert len(cluster) > 0 and len(cluster) == len(set(cluster))

        image_id_list = self.get_image_id_list_for_cluster(cluster)
        sort_index = np.argsort(image_id_list)  # return array()

        # We do not care if the bounding boxes in the cluster are already sorted, i.e., sort_index = 0, 1, 2, ...
        global_bbox_id_list_sorted = [cluster[k] for k in sort_index]

        return global_bbox_id_list_sorted

    def get_bbox_rect_list(self, global_bbox_id_list):
        """
        bbox_rect format: x1, y1, x2, y2
        :param global_bbox_id_list:
        :return:
        """
        self.assert_cluster_non_empty(global_bbox_id_list)
        return [self.get_single_bbox(x)[1:5] for x in global_bbox_id_list]

    @staticmethod
    def sorted_cluster_index_based_on_length(clusters):
        cluster_sorted_index = sorted(range(len(clusters)),
                                           key=lambda k: len(clusters[k]),
                                           reverse=True)
        return cluster_sorted_index

    def sorted_clusters_based_on_length(self, clusters):
        cluster_sorted_index = self.sorted_cluster_index_based_on_length(clusters)
        sorted_clusters = [clusters[x] for x in cluster_sorted_index]
        return sorted_clusters

    def sorted_clusters_based_on_image_id(self, clusters):
        cluster_sorted_index = sorted(
            range(len(clusters)),
            key=lambda k: min(self.get_image_id_list_for_cluster(clusters[k])),
            reverse=False)  # sort the list based on its first element

        sorted_clusters = [clusters[x] for x in cluster_sorted_index]
        return sorted_clusters

    def report_information_of_resulting_clusters(self, clusters):
        clusters = self.sorted_clusters_based_on_length(clusters)
        for idx, cluster in enumerate(clusters):
            print(f'Cluster {idx + 1}/{len(clusters)}, length: {len(cluster)}')
        print(f'Total {len(clusters)} clusters, max length= {len(clusters[0])}, '
              f'min length= {len(clusters[-1])}')

    ###################################################
    # Section: utilities for object_tracking
    ###################################################
    @staticmethod
    def count_frequency_for_item_in_list_and_sort(item_list):
        # print(f'Item list: {item_list}')

        item_unique = list(Counter(item_list).keys())  # equals to list(set(words))
        # print(f'Unique items are {item_unique}')

        # transfer dict_values() to list
        item_count = list(Counter(item_list).values())  # counts the elements' frequency
        # print(f'Count for these items are {item_count}')

        item_count_sorted_index = sorted(
            range(len(item_count)),
            key=lambda k: item_count[k],
            reverse=True
        )  # sort the list from large to small

        # result, class id sorted,
        #   item_unique[ item_count_sorted_index[0] ],
        #   item_unique[ item_count_sorted_index[1] ],
        #   ...
        #   item_unique[ item_count_sorted_index[-2] ],
        # c = [a[x] for x in b]
        item_unique_sorted = [item_unique[x] for x in item_count_sorted_index]
        item_count_sorted = [item_count[x] for x in item_count_sorted_index]
        return [item_unique_sorted, item_count_sorted]

    def estimate_majority_class_id(self, cluster):
        # class_labels = self.video_detected_bbox_all[ single_cluster, 1 ]
        if cluster is None or len(cluster) == 0:
            print('No element in the cluster.')
            sys.exit(0)

        class_labels = self.get_class_label_list_for_cluster(cluster)
        # print(f'class IDs are {class_labels}')
        class_unique_sorted, class_count_sorted = self.count_frequency_for_item_in_list_and_sort(class_labels)
        # print(f'Unique class IDs are {class_unique_sorted}')
        # print(f'Count for these class IDs are {class_count_sorted}')

        majority_class_idx = class_unique_sorted[0]
        # print(f'Majority class id is {majority_class_idx}')

        return majority_class_idx

    def estimate_cluster_label(self, clusters):
        """
        estimate the majority class lable for each cluster
        :return:
        """
        cluster_labels = []
        for idx, cluster in enumerate(clusters):
            majority_class_idx = self.estimate_majority_class_id(cluster)

            # class_list = self.get_class_label_list_for_cluster(cluster)
            # print(f'class_list = {class_list}, majority_class_idx = {majority_class_idx}')
            cluster_labels.append(majority_class_idx)
        # sys.exit(0)
        return cluster_labels

    # Section: functions used in the debugging process for object_tracking
    def clean_clusters_by_sorting_image_id(self, clusters):
        """
        Sort each cluster based on image id.
        :param clusters:
        :return:
        """
        assert len(clusters) > 0
        clusters_cleaned = [self.sort_cluster_based_on_image_id(cluster) for cluster in clusters]
        return clusters_cleaned

    def print_out_single_cluster_information(self, clusters, cluster_id):
        """
        display the information of a single cluster for debug purpose
        :param clusters: all the clusters
        :param cluster_id: the cluster id to display
        :return: nothing
        """
        print('----------------------------------------------')
        print(f'Begin to display the information for cluster {cluster_id}/{len(clusters)}')

        num_bbox = len(clusters[cluster_id])
        print(f'Total number of bbox: {num_bbox}')

        for index, global_bbox_id in enumerate(clusters[cluster_id]):
            image_id = self.get_image_id(global_bbox_id)
            bbox = self.get_single_bbox(global_bbox_id)
            class_idx, x1, y1, x2, y2 = bbox
            print(f'{index}/{num_bbox}, image_id = {image_id}, global_bbox_id = {global_bbox_id}, '
                  f'class id = {class_idx}, [{x1}, {y1}, {x2}, {y2}]')

        image_id_list = self.get_image_id_list_sorted(clusters[cluster_id])
        print(f'cluster_id = {cluster_id}, min(image_id_list) = {min(image_id_list)}, '
              f'max(image_id_list) = {max(image_id_list)}')
        print('----------------------------------------------')

    @staticmethod
    def assert_clusters_non_empty(clusters):
        """
        This function asserts if clusters are empty
        :return:
        """
        assert len(clusters) > 0, 'The number of clusters is 0, please check ....'

    @staticmethod
    def assert_cluster_member_unique(clusters):
        """
        Assert if all the cluster members are unique in the resulting clusters.
        This function can be used when a detection is not allowed to appear in more
        then one cluster.
        When there is a need to assign one detection to more than one cluster, we'd
        better copy the detection and assign a new global bbox id.
        :return:
        """
        cluster_member = []
        for cluster in clusters:
            cluster_member.extend(cluster)

        assert len(cluster_member) == len(set(cluster_member))

    @staticmethod
    def cluster_id_to_global_bbox_idx(clusters):
        """
        assign a cluster id to each of the global bbox idx, so that, we can know the
        cluster id by checking the global bbox id .
        key_from_detection or key_from_cluster
        :return:
        """
        assert len(clusters) > 0
        # [None] * len(self.video_detected_bbox_all)

        # initialize a dictionary with the same keys of self.video_detected_bbox_all,
        # and each one will map to None
        # if key_from_detection:
        #     bbox_cluster_IDs = dict.fromkeys(self.video_detected_bbox_all.keys(), None)
        # else:  # key_from_cluster
        bbox_cluster_IDs = {}
        for cluster_idx, cluster in enumerate(clusters):
            for global_bbox_id in cluster:
                # [image_id, bbox, bbox_id] -> [image_id, bbox, bbox_id, cluster_id]
                bbox_cluster_IDs[global_bbox_id] = cluster_idx
        # smrc.line.save_1d_list_to_file('bbox_cluster_IDs', bbox_cluster_IDs)

        # the following line should keep commented, as clusters do not need to cover all global bbox ids
        # assert len([x for x in bbox_cluster_IDs if x is None]) == 0

        return bbox_cluster_IDs

    def get_cluster_id_from_bbox_cluster_IDs(self, global_bbox_id):
        """
        get the cluster id through global_bbox_id
        :param global_bbox_id:
        :return:
        """
        assert len(self.bbox_cluster_IDs) > 0
        if global_bbox_id in self.bbox_cluster_IDs:
            return self.bbox_cluster_IDs[global_bbox_id]
        else:
            print(f'Note: global_bbox_id {global_bbox_id} is not assigned with any cluster ID '
                  f'in self.bbox_cluster_IDs, None is returned.')
            return None

    def _print_out_single_cluster_information(self, cluster):
        """
        display the information of a single cluster for debug purpose
        :param cluster: the cluster id to display
        :return: nothing
        """
        print('----------------------------------------------')
        # print(f'Begin to display the information for cluster {cluster_id}/{len(clusters)}')

        num_bbox = len(cluster)
        print(f'Total number of bbox: {num_bbox}')

        for index, global_bbox_id in enumerate(cluster):
            image_id = self.get_image_id(global_bbox_id)
            bbox = self.get_single_bbox(global_bbox_id)
            class_idx, x1, y1, x2, y2 = bbox
            print(f'{index}/{num_bbox}, image_id = {image_id}, global_bbox_id = {global_bbox_id}, '
                  f'class id = {class_idx}, [{x1}, {y1}, {x2}, {y2}]')

        image_id_list = self.get_image_id_list_sorted(cluster)
        print(f'min(image_id_list) = {min(image_id_list)}, '
              f'max(image_id_list) = {max(image_id_list)}')
        print('----------------------------------------------')

    def _assert_clusters_non_empty(self):
        """
        This function asserts if the self.clusters are empty
        :return:
        """
        assert len(self.clusters) > 0, 'The number of clusters in self.clusters is 0, please check ....'
        # pass

    # curve fitting section
    def get_number_of_hole_in_cluster(self, cluster):
        image_id_list = self.get_image_id_list_sorted(cluster)
        return image_id_list[-1] - image_id_list[0] + 1 - len(cluster)

    @staticmethod
    def is_unique_list(my_list):
        # assert
        return len(my_list) == len(set(my_list))

    def exist_conflict_for_members_in_single_cluster(self, cluster):
        assert len(cluster) > 0
        image_ids = self.get_image_id_list_for_cluster(cluster)
        return not(self.is_unique_list(image_ids) and self.is_unique_list(cluster))

    def recover_frame_dets(self):
        self.frame_dets = {image_path: [] for image_path in self.IMAGE_PATH_LIST}
        for global_bbox_id in self.video_detected_bbox_all.keys():
            image_path = self.IMAGE_PATH_LIST[self.get_image_id(global_bbox_id)]
            self.frame_dets[image_path].append(global_bbox_id)
        # self.LAST_IMAGE_INDEX = len(self.IMAGE_PATH_LIST) - 1
        print(f'Initialize self.frame_dets with '
              f'{len(self.video_detected_bbox_all)} detections ...')


class QueryDeprecated(Query):
    def __init__(self):
        super().__init__()

    # not used yet
    def estimate_majority_class_id_with_must_not_filter(self, cluster, ignored_class_list):
        """
        tTo find the first class in the sorted majority class list (based on frequency in descending order)
        that are not in ignored_class_list.
        If the majority class happened to be in ignored_class_list, then move to the next one that until it
        is not in ignored_class_list.
        :param cluster: a list of global_bbox_id
        :param ignored_class_list: the class list we ignore
        :return:
        """
        class_labels = self.get_class_label_list_for_cluster(cluster)
        print(f'class IDs are {class_labels}')
        class_unique_sorted, class_count_sorted = self.count_frequency_for_item_in_list_and_sort(class_labels)

        majority_class_idx = None
        for class_idx in class_unique_sorted:
            if class_idx not in ignored_class_list:
                majority_class_idx = class_idx
                break

        return majority_class_idx


class IO(Query):
    def __init__(self):
        super().__init__()

    @staticmethod
    def load_annotation_detection(image_list, image_dir, label_dir):
        video_annotation_list = []
        for image_name in image_list:
            # load the labels
            ann_path = smrc.utils.get_image_or_annotation_path(
                image_name, image_dir, label_dir,
                '.txt'
            )
            bbox_list = smrc.utils.load_bbox_from_file(ann_path)
            video_annotation_list.append(
                [image_name, bbox_list]
            )
        return video_annotation_list

    @staticmethod
    def load_json_detection(json_file, image_list, score_thd=None, nms_thd=None):
        # a detection list of the format of [class_idx, x1, y1, x2, y2, score]
        video_detection_list = json_det_to_tracking_format(
            json_file=json_file, test_image_list=image_list,
            score_thd=score_thd, nms_thd=nms_thd
        )
        return video_detection_list

    # since we need the global bbox list be initialized even without conducting tracking
    # for utilities such as html visualization, we put the initialization interface here,
    # instead of putting them in SMRCTracker
    def load_video_detection_all(self, video_detection_list):
        """load the detection list for offline object_tracking
        initialize self.video_detected_bbox_all, self.frame_dets, self.connectedness
        :param video_detection_list: a list of detection list, each detection has the format
            of [class_idx, x1, y1, x2, y2, score] or [class_idx, x1, y1, x2, y2]
        :return:
        """
        self.video_detected_bbox_all = {}
        self.frame_dets = {}  # a point represent a list [image_id, bbox, bbox_id]
        self.IMAGE_PATH_LIST = []  # for quick access the image path

        for image_id, img_detection in enumerate(video_detection_list):
            # print(img_detection)
            img_path, detection_list = img_detection

            # load image path and bbox_list
            self.IMAGE_PATH_LIST.append(img_path)
            if len(detection_list) > 0:
                id_str = len(self.video_detected_bbox_all)
                for detection in detection_list:
                    global_bbox_id = len(self.video_detected_bbox_all)
                    if len(detection) == 5:  # no detection score [class_idx, x1, y1, x2, y2]
                        bbox = list(map(int, detection))
                        # self.video_detected_bbox_all[global_bbox_id] = Detection(image_id=image_id, bbox=bbox,
                        #                                                          score=1.0)
                        self.video_detected_bbox_all[global_bbox_id] = {
                            'image_id': image_id,
                            'bbox': bbox,
                            'score': 1.0  # we assume this detection is 100% confident
                        }

                    elif len(detection) == 6:  # with detection score [class_idx, x1, y1, x2, y2, score]
                        bbox, score = list(map(int, detection[0:5])), detection[5]
                        # self.video_detected_bbox_all[global_bbox_id] = \
                        #     Detection(image_id=image_id, bbox=bbox, score=score)
                        self.video_detected_bbox_all[global_bbox_id] = {
                            'image_id': image_id,
                            'bbox': bbox,
                            'score': score  # we assume this detection is 100% confident
                        }

                # only save the key of the detections
                self.frame_dets[img_path] = list(
                    range(id_str, len(self.video_detected_bbox_all))
                )
            else:
                self.frame_dets[img_path] = []

        # self.LAST_IMAGE_INDEX = len(self.IMAGE_PATH_LIST) - 1
        print('Total number of detected bbox = %d loaded ...'
              % (len(self.video_detected_bbox_all),))

    def load_video_feature_list(self, video_feature_list):
        """Initializing features for self.video_detected_bbox_all. Note that
        the way of assigning the global bbox id is exactly the same with
        how we assign it to the detection id, so we need to make sure they have exactly
        the same order (2D list).
        :param video_feature_list:
        :return:
        """
        assert video_feature_list is not None
        count = 0
        for i, features in enumerate(video_feature_list):
            for feature_array in features:
                #
                self.video_detected_bbox_all[count]['feature'] = feature_array
                count += 1
                # print(f'self.get_feature(i) = {self.get_feature(i)}')
        print(f'Initializing {count} features for {len(self.video_detected_bbox_all)} detections done ...')
        assert count == len(self.video_detected_bbox_all)
        # print(f'len(self.video_detected_bbox_all) = {len(self.video_detected_bbox_all)}')

    def from_child_class_instance(self, child_class_instance):
        """Initialize data_hub from its child class instance.

        :param child_class_instance:
        :return:
        """
        self.video_detected_bbox_all = child_class_instance.video_detected_bbox_all
        self.clusters = child_class_instance.clusters
        self.IMAGE_PATH_LIST = child_class_instance.IMAGE_PATH_LIST

        # if need recover frame_dets, then run self.recover_frame_dets()
        self.frame_dets = child_class_instance.frame_dets
        self.bbox_cluster_IDs = {}

    def save_tracking_result_mot_format(self, result_file_name):
        assert len(self.clusters) > 0
        print(f'Generating object_tracking results in mot format to {result_file_name} ...')
        dir_name = os.path.dirname(result_file_name)
        smrc.utils.generate_dir_if_not_exist(dir_name)
        with open(result_file_name, 'w') as new_file:
            for idx, cluster in enumerate(self.clusters):
                print(f'Saving cluster {idx}, {len(cluster)} bbox ...')
                for global_bbox_id in cluster:
                    image_id = self.get_image_id(global_bbox_id)
                    bbox = self.get_single_bbox(global_bbox_id)
                    # print(f'global_bbox_id = {global_bbox_id}, bbox = {bbox} ')
                    # print(f'self.object_id_to_display = self.object_id_to_display')
                    # sys.exit(0)
                    bbox_tlwh = smrc.utils.bbox_to_tlwh(bbox)

                    # change image id and track id from 0-index to 1-index
                    # [0, 0, 1, 189, 237, 244, 278]
                    row = [image_id + 1, idx + 1] + bbox_tlwh[1:5]
                    # print(result)

                    # the score does not matter (1, or 0.5 exhibit the same evaluation results)
                    txt_line = '%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
                        row[0], row[1], row[2], row[3], row[4], row[5])
                    # txt_line = ', '.join(map(str, result))
                    # print(txt_line)
                    # sys.exit(0)
                    new_file.write(txt_line + '\n')
        new_file.close()


# class Maintain(Query):
#     def __init__(self):
#         super().__init__()


class Edit(Query):
    def __init__(self):
        super().__init__()

    ##################################
    # self maintain, we should avoid using the following methods
    ##################################

    def _assign_cluster_id_to_global_bbox_idx(self):
        """
        assign a cluster id to each of the global bbox idx, so that, we can know the
        cluster id by checking the global bbox id .
        :return:
        """
        assert len(self.clusters) > 0
        self.bbox_cluster_IDs = self.cluster_id_to_global_bbox_idx(self.clusters)

    def _delete_low_score_track(self, score_thd, criteria='mean'):
        self._assert_clusters_non_empty()
        # score_thd = self.detection_accept_score_thd
        self.clusters = self.delete_low_score_track(self.clusters, criteria=criteria, score_thd=score_thd)

    def _delete_cluster_with_length_thd(self, track_min_length_thd):
        self._assert_clusters_non_empty()
        self.clusters = self.delete_cluster_with_length_thd(
            clusters=self.clusters,
            track_min_length_thd=track_min_length_thd
        )

    ##################################
    # modify the detection
    ##################################
    def modify_single_bbox(self, global_bbox_id, bbox_new):
        self.video_detected_bbox_all[global_bbox_id]['bbox'] = bbox_new

    def class_label_to_correct(self, clusters):
        """
        # correct wrong labels inside track with majority label
        :return:
        """
        assert len(clusters) > 0
        # estimate the cluster labels if they are not done yet
        cluster_labels = self.estimate_cluster_label(clusters=clusters)
        assert len(cluster_labels) > 0

        bbox_to_modify = {}
        for cluster_id, cluster in enumerate(clusters):
            # print(f'Correcting class label for cluster {cluster_id}, {len(cluster)} boxes ..')
            cluster_label = cluster_labels[cluster_id]

            # self.print_out_single_cluster_information(clusters, cluster_id)
            for global_bbox_id in cluster:
                bbox = self.get_single_bbox(global_bbox_id)
                if bbox[0] != cluster_label:
                    print('========================================== Possible wrong class label detected ...')
                    self.print_out_single_cluster_information(clusters, cluster_id)
                    bbox_new = [cluster_label] + bbox[1:5]
                    bbox_to_modify[global_bbox_id] = bbox_new

        return bbox_to_modify

    def _correct_class_label_by_majority_voting(self, clusters):
        """
        # correct wrong labels inside track with majority label
        :return:
        """
        bbox_to_modify = self.class_label_to_correct(clusters)
        for global_bbox_id, bbox_new in bbox_to_modify.items():
            print(f'Modify global_bbox_id {global_bbox_id} in video detection list, '
                  f'old_bbox = {self.get_single_bbox(global_bbox_id)}, '
                  f'new_bbox = {bbox_new}')
            # modify the class label in detection list
            self.modify_single_bbox(global_bbox_id, bbox_new)

    def add_multiple_detections(self, detections):
        assert len(detections) > 0
        print(f'Adding detections: {len(detections)} will be added ... ')
        new_global_bbox_ids = [self.add_single_detection(image_id, bbox)
                               for image_id, bbox in detections]
        return new_global_bbox_ids

    def add_single_detection(
            self, image_id, bbox, score=1.0
    ):
        """
        Adding single detection by, for instance, curve fitting,
        to the video detection list.
        :param : a list of tuples of format (image_id, bbox), or a list of list of
            the format of [image_id, bbox]
        :return:
        """
        bbox = list(map(int, bbox))
        print(f'Adding detections: [{image_id}, {bbox}] will be added ... ')

        new_global_bbox_id = len(self.video_detected_bbox_all)
        # detection = [image_id, bbox, len(self.video_detected_bbox_all)]
        # self.video_detected_bbox_all.append(detection)
        # if len(self.score_list) > 0:
        #     self.score_list.append(1.0)

        self.video_detected_bbox_all[new_global_bbox_id] = {
            'image_id': image_id,
            'bbox': bbox,
            'score': score  # for newly add detection, we assume its score is 1.0
        }
        # self.video_detected_bbox_all[new_global_bbox_id] = Detection(
        #     image_id=image_id, bbox=bbox, score=1.0
        # )
        image_path = self.IMAGE_PATH_LIST[image_id]
        self.frame_dets[image_path] = [new_global_bbox_id]

        return new_global_bbox_id

    # post-processing
    def delete_low_score_track(self, clusters, score_thd, criteria='mean'):
        """
        Delete the tracks with low score
        :param criteria: 'mean' or 'max' using self.score
        :param score_thd:
        :param clusters:
        :return:
        """
        self.assert_clusters_non_empty(clusters)
        # if we want to remove the clusters in realtime, then have to handle if from the last one to the first one
        for i, cluster in enumerate(clusters[::-1]):
            # estimate the score for each cluster based on the criteria
            cluster_scores = [self.get_detection_score(global_bbox_id) for global_bbox_id in cluster]

            delete_flag = False
            # self.detection_accept_score_thd = 0.25
            if criteria == 'mean' and np.mean(cluster_scores) < score_thd:
                delete_flag = True

            elif criteria == 'max' and max(cluster_scores) < score_thd:
                delete_flag = True

            print(f'cluster {len(clusters) - i}/{len(clusters)}, {len(cluster)} detections, '
                  f'delete_flag = {delete_flag}, '
                  f'cluster_scores = {cluster_scores}, '
                  f'criteria={criteria} ... ')
            if delete_flag:
                clusters.remove(cluster)

        return clusters

    def delete_track_with_single_bbox_low_score(self, clusters, score_thd):
        """
        Delete the tracks with low score
        :param score_thd:
        :param clusters:
        :return:
        """
        self.assert_clusters_non_empty(clusters)

        for cluster in clusters[::-1]:
            if len(cluster) == 1 and self.get_detection_score(cluster[0]) < score_thd:
                print(f'Removing cluster [{cluster}]')
                clusters.remove(cluster)
        return clusters

    @staticmethod
    def delete_cluster_with_length_thd(clusters, track_min_length_thd=4):
        print(f'Before deleting outlier cluster, len(clusters) = {len(clusters)}')
        for idx in range(len(clusters) - 1, -1, -1):
            if len(clusters[idx]) < track_min_length_thd:
                del clusters[idx]
        print(f'After deleting outlier cluster, len(clusters) = {len(clusters)}')
        return clusters

    @staticmethod
    def remove_cluster_ids_from_clusters(clusters, cluster_id_to_delete):
        assert len(cluster_id_to_delete) > 0

        # sort by decreasing order
        cluster_id_to_delete.sort(reverse=True)

        print(f'deleting {len(cluster_id_to_delete)} clusters {cluster_id_to_delete} from '
              f'len(clusters) = {len(clusters)} ... ')

        for cluster_id in cluster_id_to_delete:
            # print(f'clusters_major to remove cluster {cluster_id}, {clusters_major[cluster_id]}')
            del clusters[cluster_id]

        print(f'deleting clusters done, len(clusters) = {len(clusters)} ... ')
        return clusters

    @staticmethod
    def cal_overlap_ratio(num_overlap, len_list1, len_list2):
        max_len = max(len_list1, len_list2)
        return num_overlap / (len_list1 + len_list2 - num_overlap), max_len

    # to be improved in the future
    def fill_in_missed_detection(
            self, clusters, dense_thd=None, class_label_to_fill=None,
            save_filled_detection_flag=None
    ):
        """
        # if save_filled_detection_flag == 'detection_file':
        #     self.save_fitted_bbox_list(fitted_bbox_list)
        # elif save_filled_detection_flag == 'detection_file_for_check':
        #     self.save_fitted_bbox_for_checking(
        #         bbox_data, fitted_bbox_list
        #     )
        #
        :param clusters:
        :param dense_thd:
        :param class_label_to_fill:
        :param save_filled_detection_flag:
        :return:
        """
        for cluster_id, cluster in enumerate(clusters):
            if len(cluster) < 3:
                continue

            num_hole = self.get_number_of_hole_in_cluster(cluster)
            print(f'Checking cluster {cluster_id}, {len(cluster)} bboxes, {num_hole} holes to fill ..')

            if dense_thd is None:
                dense_thd = 0.5

            # number of hole should be less the length of the track
            print(f'num_hole / len(cluster) = {num_hole / len(cluster)}, dense_thd {dense_thd}')
            if 0 < num_hole < len(cluster) * dense_thd:
                # if 0 < num_hole:  #
                print('Valid ...')
                print('==========================================')
                # self.print_out_single_cluster_information(clusters, cluster_id)
                fitted_bbox_list = self.fill_in_missed_detection_by_curve_fitting(
                    cluster=cluster,
                    new_class_label=class_label_to_fill
                )
                if save_filled_detection_flag == 'video_detection_list' and len(fitted_bbox_list) > 0:
                    # added_detection_list = self.video_detected_bbox_all[-len(fitted_bbox_list):]
                    # global_bbox_ids = [x[-1] for x in added_detection_list]
                    self.add_multiple_detections(fitted_bbox_list)
                    num_detection = len(self.video_detected_bbox_all)
                    global_bbox_ids = list(range(num_detection - len(fitted_bbox_list), num_detection))
                    clusters[cluster_id] += global_bbox_ids
            else:
                print('Skipping ...')

    # to be improved in the future 'save_filled_detection_flag'
    def fill_in_missed_detection_by_curve_fitting(
            self, cluster, new_class_label=None
            # save_filled_detection_flag='detection_file'
    ):
        if new_class_label is None:
            new_class_label = self.estimate_majority_class_id(cluster)

        curve_fitter = CurveFitting(image_path_list=self.IMAGE_PATH_LIST)

        bbox_data = []
        for idx, bbox_id in enumerate(cluster):

            image_id, bbox = self.get_image_id_and_bbox(bbox_id)
            # estimate the area of the bounding bbox
            bbox_data.append([image_id, bbox])
        # print(f'bbox_data = {bbox_data}')

        # [ [image_id, bbox] ] not [ [image_id, bbox_list], ... ]
        fitted_bbox_list = curve_fitter.bbox_fitting(
            bbox_data=bbox_data, class_label_to_fill=new_class_label
        )
        fitted_bbox_list = [x for x in fitted_bbox_list if x[1] is not None]
        # # len(fitted_bbox_list) can be 0
        # assert fitted_bbox_list is not None
        # if len(fitted_bbox_list) == 0:
        #     return []

        return fitted_bbox_list


# , Maintain
class DataHub(Edit, IO):
    def __init__(self):
        # super(Edit, self).__init__()
        # super(IO, self).__init__()
        super().__init__()


# class Trash:
#     def __init__(self):
#         self.video_detected_bbox_all = None
#         # a dictionary, key (global_bbox_id), value, a dict of 'image_id', 'bbox', 'score'
#         self.frame_dets = None
#         self.clusters = None
#         self.bbox_cluster_IDs = None  # {}
#         self.IMAGE_PATH_LIST = None