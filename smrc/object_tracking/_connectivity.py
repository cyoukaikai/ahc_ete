import numpy as np

import smrc.utils
from object_tracking.data_hub import Query


class Connectivity(Query):
    def __init__(self):
        super().__init__()

    def generate_empty_connectivity(self):
        """
        The keys in connectedness ambiguity confidence are identical and in the same order.
        Generate empty variable connectivity,
            connectedness, the possible connections recommended by a specific heuristic
            ambiguity: the ambiguity the detection, measured by
                if the similarity of the second recommended connection for det_i, or the second recommended
                connection for det_j is > thd, then return 0 (not sparse);
                Otherwise, return 1.
            confidence: the similarity between det_i and det_j for the first recommended connection
        :return:
        """
        # we should use keys instead of the row index.
        # connectivity = {
        #     'connectedness': [None] * len(self.video_detected_bbox_all),
        #     'ambiguity': [None] * len(self.video_detected_bbox_all),
        #     'confidence': [None] * len(self.video_detected_bbox_all)
        # }
        # dict.fromkeys(self.video_detected_bbox_all.keys(), None)
        # connectivity = {}
        # for global_bbox_id in self.video_detected_bbox_all.keys():
        #     connectivity[global_bbox_id] = {
        #         'connectedness': None,
        #         'ambiguity': None,
        #         'confidence': None
        #     }
        connectivity = {
            'connectedness': dict.fromkeys(self.video_detected_bbox_all.keys(), None),
            'ambiguity': dict.fromkeys(self.video_detected_bbox_all.keys(), None),
            'confidence': dict.fromkeys(self.video_detected_bbox_all.keys(), None)
        }
        return connectivity

    @staticmethod
    def extract_non_empty_connection(connectivity):
        # using index as the starting point of a connection
        # connections = [(i, x) for i, x in enumerate(connectivity['connectedness']) if x is not None]
        # scores = [x for x in connectivity['confidence'] if x is not None]
        # ambiguities = [x for x in connectivity['ambiguity'] if x is not None]

        # Second approach: i, the global_bbox_id
        connections = [(global_bbox_id, j) for global_bbox_id, j in connectivity['connectedness'].items()
                       if j is not None]
        scores = [connectivity['confidence'][global_bbox_id]
                  for global_bbox_id, j in connectivity['connectedness'].items() if j is not None]
        ambiguities = [connectivity['ambiguity'][global_bbox_id]
                       for global_bbox_id, j in connectivity['connectedness'].items()
                       if j is not None]
        # there may be bugs if we use the following sentence, as if we did not store the score and ambiguity
        # for a specific connection, it will be None
        # ambiguities = [ambiguity for global_bbox_id,
        # ambiguity in connectivity['ambiguity'].items() if ambiguity is not None]

        # connections, scores, ambiguities = [], [], []
        # for i in connectivity.keys():
        #     connection, score, ambiguity = connectivity[i]['connectedness'], connectivity[i]['confidence'], \
        #                                    connectivity[i]['ambiguity']
        #     if connection is not None:
        #         connections.append((i, connection))
        #
        #         if score is not None:
        #             scores.append(score)
        #         else:
        #             scores.append(0)
        #
        #         if ambiguity is not None:
        #             ambiguities.append(ambiguity)
        #         else:
        #             ambiguities.append(0)
        return connections, scores, ambiguities

    @staticmethod
    def extract_non_empty_connection_with_score_thd(connectivity, score_thd):
        print(f'Extracting non empty connections and score >= {score_thd} ...')
        connections, scores, ambiguities = [], [], []
        for global_bbox_id, j in connectivity['connectedness'].items():
            score, ambiguity = connectivity['confidence'][global_bbox_id], \
                                           connectivity['ambiguity'][global_bbox_id]
            if j is not None and score is not None and score >= score_thd:
                connections.append((global_bbox_id, j))
                scores.append(score)
                ambiguities.append(ambiguity)

        return connections, scores, ambiguities

    def extract_non_empty_connectivity_with_min_score_and_sorted(self, connectivity, score_thd=0.0):
        connections, scores, ambiguities = self.extract_non_empty_connection_with_score_thd(
            connectivity, score_thd=score_thd
        )
        # sort the connections
        sorted_index = list(np.argsort(scores))[::-1]
        sorted_connections = [connections[x] for x in sorted_index]
        sorted_scores = [scores[x] for x in sorted_index]
        ambiguities = [ambiguities[x] for x in sorted_index]

        return sorted_connections, sorted_scores, ambiguities

    @staticmethod
    def filter_out_low_confident_or_ambiguous_connection(connectivity, confidence_thd, ambiguity_thd=0):
        count = 0
        for i, x in connectivity['connectedness'].items():
            if x is not None and (connectivity['ambiguity'][i] > ambiguity_thd or
                                  connectivity['confidence'][i] < confidence_thd):
                connectivity['connectedness'][i] = None
                connectivity['confidence'][i] = None
                connectivity['ambiguity'][i] = None
                count += 1
        print(f'{count} connections are filtered out by filter_out_low_confident_or_ambiguous_connection() ...')
        return connectivity

    @staticmethod
    def report_num_of_connection(connectivity):
        valid_list = [1 for x in connectivity['connectedness'].values() if x is not None]
        num = len(valid_list)
        print(f'Number of connection is {num} .')
        return num

    @staticmethod
    def save_connectivity_to_file(connectivity, output_file_name):
        smrc.utils.save_multi_dimension_list_to_file(
            filename=output_file_name,
            list_to_save=list(
                zip(
                    list(connectivity['connectedness'].items()),
                    list(connectivity['confidence'].items()),
                    list(connectivity['ambiguity'].items())
                )
            )
        )

    @staticmethod
    def connectedness_to_clusters(connectedness):
        """
        # connectedness = connectivity['connectedness']
        V0: # used_flag = np.zeros((len(self.video_detected_bbox_all), 1))
        :param connectedness:
        :return:
        """
        # for batch clustering, we do not need the following condition to meet
        # assert len(connectedness) == len(self.video_detected_bbox_all)

        clusters = []
        used_flag = dict.fromkeys(connectedness.keys(), 0)
        # cluster_ids = np.zeros([len(self.video_detected_bbox_all),1]) * (-1)
        for global_bbox_id in connectedness.keys():
            # if this bbox has been clustered into one of the recoreded clusters.
            if used_flag[global_bbox_id] == 1:
                continue
            else:
                # # image_id, bbox, bbox_id = point
                # image_id, bbox = self.get_image_id_and_bbox(bbox_id)

                single_cluster = [global_bbox_id]
                # single_cluster.append(bbox_id)
                used_flag[global_bbox_id] = 1

                # extracting connected components
                neighboring_bbox_idx = connectedness[global_bbox_id]
                while neighboring_bbox_idx is not None:
                    single_cluster.append(neighboring_bbox_idx)
                    used_flag[neighboring_bbox_idx] = 1

                    # update the neighboring_bbox_idx
                    neighboring_bbox_idx = connectedness[neighboring_bbox_idx]

                clusters.append(single_cluster)
        print(f'{len(clusters)} clusters extracted from connectedness ...')
        return clusters

    def connectivity_to_clusters(self, connectivity):
        return self.connectedness_to_clusters(connectivity['connectedness'])

    def _extract_cluster_based_on_connectivity(self, connectivity):
        self.clusters = self.connectivity_to_clusters(connectivity)

    @staticmethod
    def connectivity_list_to_matrix(connectivity_list):
        # # each column is an independent solution
        # connectedness_matrix = np.transpose(
        #     np.array([x['connectedness'] for x in connectivity_list])
        # )
        # confidence_matrix = np.transpose(
        #     np.array([x['confidence'] for x in connectivity_list])
        # )
        # ambiguity_matrix = np.transpose(
        #     np.array([x['ambiguity'] for x in connectivity_list])
        # )
        # np.savetxt('connectedness_matrix', connectedness_matrix, fmt='%s')
        # np.savetxt('confidence_matrix', confidence_matrix, fmt='%d')
        # np.savetxt('ambiguity_matrix', ambiguity_matrix, fmt='%d')

        # we assume that each of the connectivity_list has identical and the same order of keys().
        # As every connectivity is generated by using self.generate_empty_connectivity()
        connectedness_matrix = np.transpose(
            np.array([list(x['connectedness'].values()) for x in connectivity_list])
        )
        confidence_matrix = np.transpose(
            np.array([list(x['confidence'].values()) for x in connectivity_list])
        )
        ambiguity_matrix = np.transpose(
            np.array([list(x['ambiguity'].values()) for x in connectivity_list])
        )
        # np.savetxt('connectedness_matrix', connectedness_matrix, fmt='%d')
        # np.savetxt('confidence_matrix', confidence_matrix, fmt='%f')
        # np.savetxt('confidence_matrix', ambiguity_matrix, fmt='%f')
        #
        # np.save('connectedness_matrix.npy', connectedness_matrix)
        # np.save('confidence_matrix.npy', confidence_matrix)
        # np.save('ambiguity_matrix.npy', ambiguity_matrix)

        # verify if the resulting matrices are valid or not
        # num_row, num_col = connectedness_matrix.shape

        # # Get the index of elements with value 15
        # result1 = np.where(connectedness_matrix is not None)
        # result2 = np.where(connectedness_matrix is not None)
        # result3 = np.where(connectedness_matrix is not None)
        # assert np.sum(result1) == num_row and np.sum(result2) == num_row and np.sum(result3) == num_row

        return connectedness_matrix, confidence_matrix, ambiguity_matrix

    @staticmethod
    def report_connection_diversity(connectivity1, connectivity2):
        """
        >>> a = {1: 23, 2 : 33}
        >>> b = {2:34, 1 : 44}
        >>> list(a.values())
        [23, 33]
        >>> list(b.values())
        [34, 44]
        >>> list(a.keys())
        [1, 2]
        >>> list(b.keys())
        [2, 1]
        So we should not compare connectivity1['connectedness'].values() with connectivity2['connectedness'].values().
        As the order of the key is not sorted. We can only use the key as the reference.
        :param connectivity1:
        :param connectivity2:
        :return:
        """
        # check if the two dicts have the identical keys
        assert set(connectivity1['connectedness'].keys()) == set(connectivity2['connectedness'].keys())
        # total number of different items
        num_diff = 0

        # the unique connection for connectedness1 and connectedness2, respectively
        # num_diff = diff_1 + diff_2 + diff_all_none
        diff_1, diff_2, diff_all_none = 0, 0, 0
        for global_bbox_id, i in connectivity1['connectedness'].items():
            j = connectivity2['connectedness'][global_bbox_id]
            # None == None ->  True
            if i != j:
                num_diff += 1
                if i is None and j is not None:
                    diff_2 += 1
                elif i is not None and j is None:
                    diff_1 += 1
                else:
                    diff_all_none += 1
        assert num_diff == diff_1 + diff_2 + diff_all_none

        connectedness1_non_empty = [x for x in connectivity1['connectedness'].values() if x is not None]
        connectedness2_non_empty = [x for x in connectivity2['connectedness'].values() if x is not None]
        print('==========================================================report_connection_diversity')
        print(f'Total detection: {len(connectivity1["connectedness"])}, '
              f'len(connectedness1_non_empty): {len(connectedness1_non_empty)}, '
              f'len(connectedness2_non_empty): {len(connectedness2_non_empty)}')
        print(f'Number of different connections: {num_diff} \n'
              f'only in connectedness1: {diff_1} connections \n'
              f'only in connectedness2: {diff_2} connections \n'
              f'both exist but different: {diff_all_none} connections ...')
        print('==========================================================')

    @staticmethod
    def merge_connectivity(connectivity1, connectivity2):
        """
        extend the connectivity1 with connectivity2.
        if connectivity2 has the new connections that connectivity1 do not have, then copy the connections
        :param connectivity1:
        :param connectivity2:
        :return:
        """

        assert len(connectivity1['connectedness']) == len(connectivity2['connectedness'])
        for i, j in connectivity1['connectedness'].items():
            # we require that connectivity1 and connectivity2 have the same connection if both of them are not None
            assert not(j is not None
                       and connectivity2['connectedness'][i] is not None
                       and j != connectivity2['connectedness'][i])
            # if connectivity2 has the new connections that connectivity1 do not have, then copy the connections
            if j is None and connectivity2['connectedness'][i] is not None:
                connectivity1['connectedness'][i] = connectivity2['connectedness'][i]
                connectivity1['confidence'][i] = connectivity2['confidence'][i]
                connectivity1['ambiguity'][i] = connectivity2['ambiguity'][i]
        return connectivity1


class ConnectivityDeprecated(Connectivity):
    def __init__(self):
        super().__init__()

    @staticmethod
    def update_connectivity_based_on_similarity(
            connectivity,  # both input and the output
            points_prev_image, points_current_image,
            similarity_matrix, min_similarity_to_connect=1e-5
    ):
        """
        update the connectivity based on the estimated similarity.
        :param connectivity: a dict includes
                connectivity['connectedness']
                connectivity['confidence']
        :param points_prev_image:
        :param points_current_image:
        :param similarity_matrix: similarity value should between [0, 1]
        :param min_similarity_to_connect:
        :return:
        """
        assert len(points_prev_image) > 0 and len(points_current_image) > 0

        # check if the size of similarity_matrix > 0, and the value of similarity_matrix
        # in the correct range [0,1]
        # similarity is used to generate recommendation level
        assert similarity_matrix.size > 0 and np.amax(similarity_matrix) <= 1
        # np.amin(similarity_matrix) >= 0  (for cosine similarity, the value could be negative
        # print(f'points_prev_image={points_prev_image}, points_current_image = {points_current_image} ...')

        # # index of the maximum value for each row
        # max_values_index_column = np.argmax(similarity_matrix, axis=0)  # return array, e.g., array([0, 0, 0])
        # # index of the maximum value for columns
        # max_values_index_row = np.argmax(similarity_matrix, axis=1)
        #
        # # print('max_values_index_column.shape = ', max_values_index_column.shape )
        # # print('max_values_index_row.shape = ', max_values_index_row.shape)

        # sorts along first axis (down) from small to large
        column_sort_indices = np.argsort(similarity_matrix, axis=0)
        # sorts along second axis (cross) from small to large
        row_sort_indices = np.argsort(similarity_matrix, axis=1)

        # print(f'points_prev_image = {points_prev_image}')
        # print(f'points_current_image = {points_current_image}')
        for i, bbox_idx_prev in enumerate(points_prev_image):
            # image_id_tmp, bbox_tmp, bbox_id_tmp = point_i  # [image_id, bbox, bbox_id]
            # print(f'{self.IMAGE_PATH_LIST[image_id_tmp]}, {bbox_tmp}')
            # j = max_values_index_row[i]
            j = row_sort_indices[i, -1]
            # I = max_values_index_column[j]
            # ii = column_sort_indices[-1, jj]
            # the max value of the distance should be larger than min_similarity_to_connect
            # max_values_index_column[j]
            if column_sort_indices[-1, j] == i and \
                    similarity_matrix[i, j] > min_similarity_to_connect:

                bbox_idx_next = points_current_image[j]

                # connect the two bboxes
                connectivity['connectedness'][bbox_idx_prev] = bbox_idx_next
                connectivity['confidence'][bbox_idx_prev] = similarity_matrix[i, j]

                second_largest_similarity_values = [0]
                # row direction
                if similarity_matrix.shape[1] > 1 and similarity_matrix[i, row_sort_indices[i, -2]] > 0:
                    second_largest_similarity_values.append(
                        similarity_matrix[i, row_sort_indices[i, -2]]
                    )
                # column direction, update the second largest element
                if similarity_matrix.shape[0] > 1 and similarity_matrix[column_sort_indices[-2, j], j] > 0:
                    second_largest_similarity_values.append(
                        similarity_matrix[column_sort_indices[-2, j], j]
                    )

                connectivity['ambiguity'][bbox_idx_prev] = max(second_largest_similarity_values)
        return connectivity
    #
    # @staticmethod
    # def update_connectivity_based_on_dissimilarity(
    #         connectivity,  # both input and the output
    #         points_prev_image, points_current_image,
    #         dissimilarity_matrix,
    #         max_dissimilarity_to_connect=1
    # ):
    #
    #     # check if the size of dissimilarity_matrix > 0, and the value of dissimilarity_matrix
    #     # in the correct range [0,1]
    #     # dissimilarity_matrix is used to generate recommendation level
    #     assert dissimilarity_matrix.size > 0 and np.amin(dissimilarity_matrix) >= 0 and \
    #            np.amax(dissimilarity_matrix) <= 1
    #     assert len(points_prev_image) > 0 and len(points_current_image) > 0
    #
    #     # find the minimum value within each column, and each row
    #     min_values_index_column = np.argmin(dissimilarity_matrix, axis=0)  # returna array, e.g., array([0, 0, 0])
    #     min_values_index_row = np.argmin(dissimilarity_matrix, axis=1)
    #     # print('min_values_index_column.shape = ', min_values_index_column.shape )
    #     # print('min_values_index_row.shape = ', min_values_index_row.shape)
    #
    #     # update the connectivity based on the estimated distance.
    #     # print(f'points_prev_image = {points_prev_image}')
    #     # print(f'points_current_image = {points_current_image}')
    #     for i, bbox_idx_prev in enumerate(points_prev_image):
    #         # image_id_tmp, bbox_tmp, bbox_id_tmp = point_i  # [image_id, bbox, bbox_id]
    #         # print(f'{self.IMAGE_PATH_LIST[image_id_tmp]}, {bbox_tmp}')
    #         j = min_values_index_row[i]
    #         # the min value of the distance should be smaller than inf_value
    #         # print(f'IoU = {dist_matrix[i, j]}')
    #         if min_values_index_column[j] == i and \
    #                 dissimilarity_matrix[i, j] < max_dissimilarity_to_connect:
    #             bbox_idx_next = points_current_image[j]
    #
    #             # connect the two bboxes
    #             connectivity['connectedness'][bbox_idx_prev] = bbox_idx_next
    #             # transfer the dist to similarity
    #             connectivity['confidence'][bbox_idx_prev] = 1 - dissimilarity_matrix[i, j]
    #
    #     return connectivity
    #
    # def update_connectivity_based_on_dist(
    #         self, connectivity,  # both input and the output
    #         points_prev_image, points_current_image,
    #         dist_matrix,
    #         max_distance_to_connect
    # ):
    #
    #     assert len(points_prev_image) > 0 and len(points_current_image) > 0
    #
    #     # find the minimum value within each column, and each row
    #     min_values_index_column = np.argmin(dist_matrix, axis=0)  # returna array, e.g., array([0, 0, 0])
    #     min_values_index_row = np.argmin(dist_matrix, axis=1)
    #     # print('min_values_index_column.shape = ', min_values_index_column.shape )
    #     # print('min_values_index_row.shape = ', min_values_index_row.shape)
    #
    #     # update the connectivity based on the estimated distance.
    #     # print(f'points_prev_image = {points_prev_image}')
    #     # print(f'points_current_image = {points_current_image}')
    #     for i, bbox_idx_prev in enumerate(points_prev_image):
    #         # image_id_tmp, bbox_tmp, bbox_id_tmp = point_i  # [image_id, bbox, bbox_id]
    #         # print(f'{self.IMAGE_PATH_LIST[image_id_tmp]}, {bbox_tmp}')
    #         j = min_values_index_row[i]
    #         # the min value of the distance should be smaller than inf_value
    #         # print(f'IoU = {dist_matrix[i, j]}')
    #         if min_values_index_column[j] == i and dist_matrix[i, j] < max_distance_to_connect:
    #             bbox_idx_next = points_current_image[j]
    #
    #             # connect the two bboxes
    #             connectivity['connectedness'][bbox_idx_prev] = bbox_idx_next
    #             # transfer the dist to similarity
    #             connectivity['confidence'][bbox_idx_prev] = self.dist_to_confidence(
    #                 dist_matrix[i, j], max_distance_to_connect
    #             )
    #             # 1 - dist_matrix[i, j] is not accurate, as dist_matrix[i, j] > max_dist_to_connect
    #
    #     return connectivity

    # def update_connectivity_iou_similarity(
    #         self, connectivity,
    #         points_prev_image, points_current_image,
    #         iou_low_thd
    # ):
    #     """
    #     Expert for IoU based connecting.
    #     This function is used  not only in IoU tracker, but also in other trackers.
    #     connect the same object in two frames incrementally
    #     a point represent a list [image_id, bbox, bbox_id]
    #     :param connectivity:
    #     :param points_prev_image:
    #     :param points_current_image:
    #     :param iou_low_thd:
    #     :return:
    #     """
    #
    #     if len(points_prev_image) == 0 or len(points_current_image) == 0:
    #         return connectivity
    #     else:
    #         similarity_matrix = self.estimate_iou_matrix(
    #             points_prev_image, points_current_image
    #         )
    #         # print(f'similarity_matrix = {similarity_matrix}')
    #         # print('dist.shape = ', dist.shape)
    #
    #         return self.update_connectivity_based_on_similarity(
    #             connectivity, points_prev_image, points_current_image,
    #             similarity_matrix, min_similarity_to_connect=iou_low_thd
    #         )

    # def merge_connectivity_list_not_finished(self, connectivity_list):
    #     resulting_connectivity = self.generate_empty_connectivity()
    #     # connectedness_list = [connectivity['connectedness'] for connectivity in connectivity_list]
    #     # ambiguity_list = [connectivity['ambiguity'] for connectivity in connectivity_list]
    #     # confidence_list = [connectivity['confidence'] for connectivity in connectivity_list]
    #
    #     # we assume that each of the connectivity_list has identical and the same order of keys().
    #     # As every connectivity is generated by using self.generate_empty_connectivity()
    #     connectedness_matrix = np.transpose(
    #         np.array([list(x['connectedness'].values()) for x in connectivity_list])
    #     )
    #     confidence_matrix = np.transpose(
    #         np.array([list(x['confidence'].values()) for x in connectivity_list])
    #     )
    #     ambiguity_matrix = np.transpose(
    #         np.array([list(x['ambiguity'].values()) for x in connectivity_list])
    #     )
    #     np.savetxt('connectedness_matrix', connectedness_matrix, fmt='%d')
    #     np.savetxt('confidence_matrix', confidence_matrix, fmt='%f')
    #     np.savetxt('confidence_matrix', ambiguity_matrix, fmt='%f')
    #
    #     num_row, num_col = connectedness_matrix.shape
    #
    #     # Get the index of elements with value 15
    #     result1 = np.where(connectedness_matrix is not None)
    #     result2 = np.where(connectedness_matrix is not None)
    #     result3 = np.where(connectedness_matrix is not None)
    #     assert np.sum(result1) == num_row and np.sum(result2) == num_row and np.sum(result3) == num_row
    #
    #     # # self.connectedness = []
    #     # for i in range(num_row):
    #     #
    #     #     voting = connectedness_matrix[i, :]
    #     #     resulting_connectivity['connectedness'] = voting[]
    #     #     unique_vote = list(set(list(voting)))
    #     #     if len(unique_vote) == 1 and unique_vote[0] is not None:
    #     #         self.connectedness[i] = unique_vote[0]
    #     #
    #     #     # weights = confidence_matrix[i, :]


    # def merge_connectivity_list_not_finished(self, connectivity_list):
    #     resulting_connectivity = self.generate_empty_connectivity()
    #     # connectedness_list = [connectivity['connectedness'] for connectivity in connectivity_list]
    #     # ambiguity_list = [connectivity['ambiguity'] for connectivity in connectivity_list]
    #     # confidence_list = [connectivity['confidence'] for connectivity in connectivity_list]
    #
    #     # we assume that each of the connectivity_list has identical and the same order of keys().
    #     # As every connectivity is generated by using self.generate_empty_connectivity()
    #     connectedness_matrix = np.transpose(
    #         np.array([list(x['connectedness'].values()) for x in connectivity_list])
    #     )
    #     confidence_matrix = np.transpose(
    #         np.array([list(x['confidence'].values()) for x in connectivity_list])
    #     )
    #     ambiguity_matrix = np.transpose(
    #         np.array([list(x['ambiguity'].values()) for x in connectivity_list])
    #     )
    #     np.savetxt('connectedness_matrix', connectedness_matrix, fmt='%d')
    #     np.savetxt('confidence_matrix', confidence_matrix, fmt='%f')
    #     np.savetxt('confidence_matrix', ambiguity_matrix, fmt='%f')
    #
    #     num_row, num_col = connectedness_matrix.shape
    #
    #     # Get the index of elements with value 15
    #     result1 = np.where(connectedness_matrix is not None)
    #     result2 = np.where(connectedness_matrix is not None)
    #     result3 = np.where(connectedness_matrix is not None)
    #     assert np.sum(result1) == num_row and np.sum(result2) == num_row and np.sum(result3) == num_row
    #
    #     # # self.connectedness = []
    #     # for i in range(num_row):
    #     #
    #     #     voting = connectedness_matrix[i, :]
    #     #     resulting_connectivity['connectedness'] = voting[]
    #     #     unique_vote = list(set(list(voting)))
    #     #     if len(unique_vote) == 1 and unique_vote[0] is not None:
    #     #         self.connectedness[i] = unique_vote[0]
    #     #
    #     #     # weights = confidence_matrix[i, :]
