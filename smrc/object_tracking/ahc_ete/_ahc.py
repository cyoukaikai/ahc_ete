############################
# AHC based tracking
##########################

import copy
from heapq import heapify, heappop, heappush
import numpy as np
from tqdm import tqdm
import time

from sklearn.cluster import _hierarchical_fast as _hierarchical
from sklearn.utils._fast_dict import IntFloatDict

from object_tracking.data_hub import Query
from object_tracking.ahc_ete.standard_ahc import AHC

"""
The original merging methods, i.e., 
    'complete': _hierarchical.max_merge,
    'average': _hierarchical.average_merge,
    'single': _hierarchical.single_merge,
are used 'union' strategy to update the can-link list, i.e., if cluster A can link with C, D and cluster B 
can link with C, E, the cluster (A, B) can link with C, D and E. 
The 'union' strategy is designed for handling sparse connectivity in which we are only aware of the local information 
thatA can link with C, D, but no information for whether A can link with E because E is far from A.

However, for object tracking, we have to reflect the spatial-temporal constraints during the initialization of the 
can-link set. So if "A can link with C, D, ..." and "E" is not in the list, then E should never be connected with a 
cluster of which A is a member. Thus, I should use the 'intersection' strategy,
i.e., if cluster A can link with C, D and cluster B can link with C, E, the cluster
(A, B) can only link with C, not D or E.
My implementations are:
    'complete': _hierarchical.max_merge_tracking,
    'average': _hierarchical.average_merge_tracking,
    'single': _hierarchical.single_merge_tracking,
    
If we set the cannot link position in pdist to BIG_VALUE (e.g., 1e+5), then using the
original python AHC (only limited to complete and average linkage) can get the same results and 
run much faster than my implementation. 
But the "single" linkage with spatial-temporal constraints is not available for the standard ahc.

Moreover, the standard ahc assume the cluster distance can be directly derived from pdist.
However, the distance of the two tracks can not always be derived from the precomputed pdist,
e.g., Temporal distance, Kalman Filter distance.
Estimating the pdist for combined (dynamic) distance metrics in advance and then conduct the 
standard ahc is impossible in some case.
If I used the standard ahc, I cannot use one major distance metric to guide the merging and 
other multiple distance metrics as filter to disable a merging. 
That is, once the recommended merging (the closest pair of clusters based on
the major distance metric) is not valid, then the merging is skipped. 

This is the reason I implemented ahc by myself.
"""

#####################################################
# the definition of the linkage choice
####################################################
linkage_choices = {
    'complete': _hierarchical.max_merge_tracking,
    'average': _hierarchical.average_merge_tracking,
    'single': _hierarchical.single_merge_tracking
}

CANNOT_LINK_DIST = 1e+6
INCLUDE_SAME_CLUSTER_DET = False


class AhcLinkage:

    @staticmethod
    def _set_linkage_choice(merge_linkage):
        try:
            join_func = linkage_choices[merge_linkage]
        except KeyError:
            raise ValueError(
                'Unknown linkage option, linkage should be one '
                'of %s, but %s was given' % (linkage_choices.keys(), merge_linkage))
        return join_func

    @staticmethod
    def assert_linkage(linkage):
        assert linkage in ['complete', 'average', 'single']


class AhcLinkageDeprecated(AhcLinkage):
    ###############################################
    # My implementation of the max_merge and average_merge using the 'intersection' strategy.
    # They have passed the tests, but they are slow than
    # the built-in C implementation. So they should not be used.
    ###############################################
    @staticmethod
    def max_merge(a, b, mask, n_a, n_b):
        """Merge two IntFloatDicts with the max strategy: when the same key is
        present in the two dicts, the max of the two values is used.

        Parameters
        ==========
        a, b : IntFloatDict object
            The IntFloatDicts to merge
        mask : ndarray array of dtype integer and of dimension 1
            a mask for keys to ignore: if not mask[key] the corresponding key
            is skipped in the output dictionary
        n_a, n_b : float
            n_a and n_b are weights for a and b for the merge strategy.
            They are not used in the case of a max merge.

        IntFloatDicts: format
        keys = d.to_arrays()[0]  # 1, value
        for key in keys:
            print(f'key = {key}, value = {d[key]}')

        Returns
        =======
        out : IntFloatDict object
            The IntFloatDict resulting from the merge
        """
        out_obj_keys, out_obj_values = [], []
        keys_a, keys_b = list(a.to_arrays()[0]), list(b.to_arrays()[0])

        if len(keys_a) == 0:
            return None
        else:
            for key in keys_a:
                if mask[key] and key in keys_b:
                    out_obj_keys.append(key)
                    out_obj_values.append(max(a[key], b[key]))
            # print(f'len(out_obj_keys) = {len(out_obj_keys)} ... ')
            # print(f'out_obj_values = {out_obj_values} ... ')
            # A[i] = IntFloatDict(np.array(row, dtype=np.intp),
            #                     np.array(data, dtype=np.float64))  # fast dict

            out_obj = IntFloatDict(np.array(out_obj_keys, dtype=np.intp),
                                   np.array(out_obj_values, dtype=np.float64))
            return out_obj

    @staticmethod
    def average_merge(a, b, mask, n_a, n_b):
        n_out = n_a + n_b
        out_obj_keys, out_obj_values = [], []
        # keys_all = list(a.keys()) + list(b.keys())
        # keys_all = [key for key in a] + [key for key in b]  # does not work
        keys_a, keys_b = list(a.to_arrays()[0]), list(b.to_arrays()[0])
        keys_all = set(keys_a + keys_b)

        if len(keys_all) == 0:
            return None
        else:
            for key in keys_all:
                if key in keys_a and mask[key] and key in keys_b:
                    out_obj_keys.append(key)
                    out_obj_values.append(
                        (n_a * a[key] + n_b * b[key]) / n_out
                    )

            out_obj = IntFloatDict(np.array(out_obj_keys), np.array(out_obj_values))
            return out_obj


class AhcMetric:
    @staticmethod
    def cluster_dist_from_pdist_rows_cols(pdist, rows, cols, linkage):
        # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
        dist_matrix = pdist[np.ix_(rows, cols)]
        return AhcMetric.cluster_distance_by_linkage(
            dist_matrix_or_list=dist_matrix,
            linkage=linkage
        )

    @staticmethod
    def cluster_distance_by_linkage(dist_matrix_or_list, linkage):
        """Used to support the linkage operation from the sub-matrix of pdist.
        linkage.find('complete') > -1:
        linkage.find('single') > -1:
        :param linkage:
        :param dist_matrix_or_list: sub-matrix of pdist extracted from the given rows and cols.
        :return:
        """
        AhcLinkage.assert_linkage(linkage)
        if linkage == 'complete':
            return np.max(dist_matrix_or_list)
        elif linkage == 'average':
            return np.mean(dist_matrix_or_list)
        else:  # linkage == 'single'
            return np.min(dist_matrix_or_list)


class HeapCluster:
    ###############################################
    # heap cluster operations
    ###############################################

    @staticmethod
    def heap_cluster_global_bbox_id_list(heap_cluster):
        return [x[1] for x in heap_cluster]

    @staticmethod
    def heap_cluster_image_id_list(heap_cluster):
        return [x[0] for x in heap_cluster]

    @staticmethod
    def heap_cluster_sorted_image_id_list(heap_cluster):
        return sorted([x[0] for x in heap_cluster])

    @staticmethod
    def remove_global_bbox_list_from_heap_cluster(
            heap_cluster, to_remove_global_bbox_id_list
    ):
        heap_cluster_new = []
        for image_id, global_bbox_id in heap_cluster:
            if global_bbox_id not in to_remove_global_bbox_id_list:
                heappush(heap_cluster_new, (image_id, global_bbox_id))
        return heap_cluster_new

    @staticmethod
    def heap_clusters_to_clusters(heap_clusters):
        """The resulting clusters by heap_clusters_to_clusters are
        sorted by image id for each cluster in default as we use heappop method.
        heap_clusters is a dict, key is the cluster_id, value is a heap_cluster
        :param heap_clusters:
        :return:
        """
        result_clusters = []
        # heap_clusters.copy() is a shallow copy and will modify the heap_clusters
        tmp_clusters = copy.deepcopy(heap_clusters)
        for key, heap_cluster in tmp_clusters.items():
            cluster = []
            while heap_cluster:
                global_bbox_id = heappop(heap_cluster)[1]
                cluster.append(global_bbox_id)
            result_clusters.append(cluster)
        return result_clusters


class SequenceDivision(Query):
    """
    Functionality used to divide the image sequence to sub-sequences.
    """

    def by_frame_id_sequence(self, frame_id_sequence):
        num_sub_seq = len(frame_id_sequence)
        num_frames = np.sum([len(x) for x in frame_id_sequence])
        D = []
        # divide image sequences into sub sequences, each with 256 frames except the last one
        print(f'Dividing image sequences ({num_frames} frames) into {num_sub_seq} sub sequences ...')
        with tqdm(total=len(frame_id_sequence)) as pbar:
            for seq_id, sub_sequence in enumerate(frame_id_sequence):
                pbar.set_description(f'Processing sub sequences {seq_id}/{num_sub_seq - 1}  ...')

                # extract the corresponding detections for each sub-sequence
                D_seq_id = []
                for image_id in sub_sequence:
                    global_bbox_ids = self.frame_dets[self.IMAGE_PATH_LIST[image_id]]
                    D_seq_id += global_bbox_ids
                D.append(D_seq_id)
                pbar.update(1)
        # print(f'Dividing image sequences done ...')
        return D

    def equal_split_frame_id(self, num_frame_per_sub_seq=None, num_sub_seq=None):
        """
        Splitting the image sequence equally based the frame ID only.
        :return:
        """
        num_frames = len(self.IMAGE_PATH_LIST)

        assert num_frame_per_sub_seq is not None or num_sub_seq is not None and not \
            (num_frame_per_sub_seq is not None and num_sub_seq is not None)
        if num_frame_per_sub_seq is not None:
            num_sub_seq = int(np.ceil(num_frames / num_frame_per_sub_seq))
        else:
            num_frame_per_sub_seq = int(np.ceil(num_frames / num_sub_seq))
        # there should be at least one frame for each subsequence
        assert num_frame_per_sub_seq > 0

        frame_id_sequence = []
        for seq_id in range(num_sub_seq):
            # print(f'Processing sub sequences {seq_id}/{num_batch - 1}  ...')
            min_frame_id, max_frame_id = num_frame_per_sub_seq * seq_id, min(num_frame_per_sub_seq * (seq_id + 1), num_frames)
            # extract the frame ids for each sequence
            frame_id_sequence.append(
                list(range(min_frame_id, max_frame_id))
            )
        self._assert_valid_division(frame_id_sequence)
        return self.by_frame_id_sequence(frame_id_sequence)

    def roughly_equal_split_det_num(self, max_det_num_per_sub_seq):
        """
        Splitting the image sequence equally based the batch size of the detection for each sub-sequence.
        We will obtain at least two non-overlapping subsequences.
        :return:
        """
        # if batch_det_num is None:
        #     batch_det_num = self.batch_det_num
        batch_size = int(round(len(self.IMAGE_PATH_LIST) /
                               (len(self.video_detected_bbox_all) / max_det_num_per_sub_seq)))
        if batch_size >= len(self.IMAGE_PATH_LIST):
            return self.equal_split_frame_id(num_sub_seq=2)
        else:
            return self.equal_split_frame_id(num_frame_per_sub_seq=batch_size)

    # def equally_split_det_num(self, batch_det_num):
    #     """ Not done yet, I do not need this function.
    #     Splitting the image sequence equally based the batch size of the detection for each sub-sequence.
    #     :return:
    #     """
    #     # if batch_det_num is None:
    #     #     batch_det_num = self.batch_det_num
    #     num_det = len(self.video_detected_bbox_all)
    #     num_batch = int(np.ceil(num_det / batch_det_num))
    #
    #     frame_id_sequence = []
    #     for seq_id in range(num_batch):
    #         print(f'Processing sub sequences {seq_id}/{num_batch - 1}  ...')
    #         min_det_id, max_det_id = batch_det_num * seq_id, min(batch_det_num * (seq_id + 1) - 1, num_det-1)
    #         ##############################################################
    #         # we should transfer the det id to global bbox id in tuture
    #         ###########################################################
    #         min_frame_id, max_frame_id = self.get_image_id(min_det_id), self.get_image_id(max_det_id)
    #         # extract the frame ids for each sequence
    #         frame_id_sequence.append(
    #             list(range(min_frame_id, max_frame_id))
    #             # Current sequence: [min_frame_id, max_frame_id), next sequence
    #             # [min_frame_id_new, max_frame_id_new)], where min_frame_id_new = max_frame_id
    #             # Here we use max_frame_id not max_frame_id + 1 because the max_frame_id is included in the
    #             # next sequence.
    #         )
    #     return self.by_frame_id_sequence(frame_id_sequence)

    def _assert_valid_division(self, frame_id_sequence):
        image_ids = []
        for sequence in frame_id_sequence:
            image_ids.extend(sequence)

        divided_image_id_set = set(image_ids)
        assert len(image_ids) == len(divided_image_id_set), \
            f'{len(image_ids)} frame IDs found in divided sequence, ' \
            f'but {len(divided_image_id_set)} unique ones found.'

        assert len(self.IMAGE_PATH_LIST) == len(divided_image_id_set), \
            f'{len(divided_image_id_set)} frame IDs found in divided sequence, ' \
            f' but {len(self.IMAGE_PATH_LIST)} frames in the whole image sequence.'

    def auto_divide_sequence(self, num_frame_per_sub_seq=None, max_det_num_per_sub_seq=None):
        #######################################
        # older version backup
        #######################################
        # if len(self.video_detected_bbox_all) > self.batch_det_num:
        #     # print(f'Conduct object_tracking for every two image sequences ...')
        #     D = self.roughly_equal_split_det_num(max_det_num_per_sub_seq=self.batch_det_num)
        # else:
        #     D = self.equal_split_frame_id(num_sub_seq=2)

        assert num_frame_per_sub_seq is not None or max_det_num_per_sub_seq is not None

        if num_frame_per_sub_seq is not None:
            assert isinstance(num_frame_per_sub_seq, int) and num_frame_per_sub_seq > 0
            D = self.equal_split_frame_id(num_frame_per_sub_seq=num_frame_per_sub_seq)
        else:
            assert isinstance(max_det_num_per_sub_seq, int) and max_det_num_per_sub_seq > 0
            D = self.roughly_equal_split_det_num(max_det_num_per_sub_seq=max_det_num_per_sub_seq)
        return D

    def estimate_num_frame_per_sub_seq(self, num_sub_seq):
        num_frames = len(self.IMAGE_PATH_LIST)
        num_frame_per_sub_seq = int(np.ceil(num_frames / num_sub_seq))
        return num_frame_per_sub_seq


class AHCTrackerUtility:

    def _is_valid_merging(self, heap_cluster1, heap_cluster2, **kwargs):
        """Check if two merging are valid based on additional constraints, return true in default
        (i.e., no further checking). Any customized constraints (kf dist, etc.) should be implemented in subclasses.
        :param heap_cluster1:
        :param heap_cluster2:
        :return:
        """
        raise NotImplementedError

    @staticmethod
    def get_complete_node_list(parent, n_samples, used_node):
        # return array of integers. Note that get the cluster information in CLUSTERS
        labels = _hierarchical.hc_get_heads(parent, copy=True)
        # copy to avoid holding a reference on the original array
        # the parent of the id > n_samples are just initialized with themselves
        labels = np.copy(labels[:n_samples])
        labels = list(np.unique(labels))

        # # remove false positives
        nodes = [x for x in labels if used_node[x]]
        return nodes

    @staticmethod
    def hc_get_head(parents, node):
        parent = parents[node]
        while parent != node:
            node = parent
            parent = parents[node]
        return parent

    @staticmethod
    def get_leaves(children, n_leaves, node_id):
        return _hierarchical._hc_get_descendent(node_id, children, n_leaves)

    @staticmethod
    def build_reference(global_bbox_id_list):
        DetID2GlobalBboxID_Array = np.array([x for x in global_bbox_id_list])
        GlobalBboxID2DetID_Dict = {x: i for i, x in enumerate(list(DetID2GlobalBboxID_Array))}
        return DetID2GlobalBboxID_Array, GlobalBboxID2DetID_Dict


def fill_in_empty_fast_dict(A_, nodes_):
    for i_ in nodes_:  # A[8221].to_arrays()[0] = -1
        A_[i_] = IntFloatDict(np.array([-1], dtype=np.intp),
                              np.array([-1], dtype=np.float64))


def fast_dict_remove_fill_in(A_, nodes_):
    for i_ in nodes_:
        # NotImplementedError: Subscript deletion not supported by sklearn.utils._fast_dict.IntFloatDict if
        # del A[n_k][-1]
        # IntFloatDict conduct insert sort, Out[27]: array([-21,  -1,   3,  21])
        keys_ = A_[i_].to_arrays()[0][1:]
        values_ = A_[i_].to_arrays()[1][1:]
        if len(keys_) > 0:
            A_[i_] = IntFloatDict(np.array(keys_, dtype=np.intp),
                                  np.array(values_, dtype=np.float64))
        else:
            A_[i_] = None  # A_ is ndarray, thus we can not do del A_[i_]


class AHCTracker(AhcLinkage, AHCTrackerUtility, SequenceDivision, HeapCluster):
    """
    Handling long image sequence.
    We assume all the detections can be load into memory for offline object tracking even we divide the sequence
    into sub sequences if the number of detections is too large so that generating pdist runs out of memory.
    So this class can not used for online tracking for videos with infinite length.
    """

    def __init__(self):
        super().__init__()
        self.used_node_ = None
        self.parent_ = None
        self.children_ = None
        self.distances_ = None
        self.clusters_ = None

        # pdist_ is only for local usage
        self.pdist_ = None
        self.n_samples_ = None

        self.batch_det_num = 3000

    def init_ahc_private_variable(self):
        """
        Initialize the private variables used in ahc tracking
        :return:
        """
        self.n_samples_ = len(self.video_detected_bbox_all)
        n_nodes = 2 * self.n_samples_ - 1

        self.parent_ = np.arange(n_nodes, dtype=np.intp)
        self.used_node_ = np.ones(n_nodes, dtype=np.intp)
        self.children_ = []
        self.distances_ = np.empty(n_nodes - self.n_samples_)
        self.pdist_ = None  # no need to touch

        self.clusters_ = self.init_heap_clusters(self.video_detected_bbox_all.keys())

        # The following code is wrong.
        # for global_bbox_id in self.video_detected_bbox_all.keys():
        #     self.clusters_[global_bbox_id] = [
        #         (self.get_image_id(global_bbox_id),
        #          global_bbox_id)]  # heap sorted by first element for the tuples in the list, but
        #     heapify(self.clusters_[global_bbox_id])

    def init_heap_clusters(self, global_bbox_id_list):
        clusters = {}
        # Saving the clusters as heap structure does not help, as we need to access both the first
        # element and the last element for the cluster, while heap is only convenient for accessing the first
        # element.
        # Note that clusters[i][0] is the smallest item, but clusters[i][-1] is not the largest value

        for i, global_bbox_id in enumerate(global_bbox_id_list):
            # https://docs.python.org/3.0/library/heapq.html
            clusters[i] = [(self.get_image_id(global_bbox_id), global_bbox_id)]
            heapify(clusters[i])
        return clusters

    def _is_valid_merging(self, heap_cluster1, heap_cluster2, **kwargs):
        """Check if two merging are valid based on additional constraints, return true in default
        (i.e., no further checking). Any customized constraints (kf dist, etc.) should be implemented in subclasses.
        :param heap_cluster1:
        :param heap_cluster2:
        :return:
        """
        return True

    def _head_from_global_bbox_id(self, global_bbox_id):
        return self.hc_get_head(parents=self.parent_, node=global_bbox_id)

    def _hc_get_unique_heads(self, global_bbox_id_list):
        return list(set([self._head_from_global_bbox_id(x) for x in global_bbox_id_list]))

    def _computer_pdist(self, global_bbox_id_list, with_st_const):
        raise NotImplementedError

    def _cluster_dist_to_edge(self, cluster_pdist_dict):
        """Need to conduct test by test it against AHCTracker function AgglomerativeTracking
        cluster_pdist_dict[n_k].append([n_l, dist])
        :param cluster_pdist_dict: key, node_k, value, node_l, and the dist of (node_k, node_l)
        :return:
        """
        n_samples = self.n_samples_
        n_nodes = 2 * n_samples - 1  # batch size
        A = np.empty(n_nodes, dtype=object)
        for i, node_dist_list in cluster_pdist_dict.items():
            if len(node_dist_list) == 0:
                A[i] = 0
            else:
                # 0 is the row, 1 is the dist
                A[i] = IntFloatDict(np.array([x[0] for x in node_dist_list], dtype=np.intp),
                                    np.array([x[1] for x in node_dist_list], dtype=np.float64))  # fast dict
        return A

    def _subsequence_tracking(self, inertia, A, distance_threshold, linkage, filter_config=None, **kwargs):
        if len(inertia) == 0: return

        join_func = self._set_linkage_choice(linkage)
        n_samples = self.n_samples_  # batch size num_clusters
        n_nodes = 2 * n_samples - 1  # batch size
        # step 2: recursive merge loop  for k in range(2 * n_samples - len(self.clusters_), n_nodes):

        for k in range(2 * n_samples - len(self.clusters_), n_nodes):
            i, j, edge = None, None, None
            while True:
                if len(inertia) == 0:
                    break

                # find the closest pair of tracks based on the given distance measure
                edge = heappop(inertia)
                i = edge.a  # index of the row, not the global bbox id
                j = edge.b

                # break if there is no valid candidate
                if edge.weight > distance_threshold:
                    break

                # if these nodes are not frozen and not in soft cannot links
                if self.used_node_[i] and self.used_node_[j]:
                    if self._is_valid_merging(
                            self.clusters_[i], self.clusters_[j], filter_config=filter_config,
                            **kwargs
                    ):
                        break

            if len(inertia) == 0 or edge.weight > distance_threshold:
                break

            if (k - n_samples) % 100 == 0:
                print(f'{k - n_samples} th merging, len(inertia) = {len(inertia)}, '
                      f'len(clusters) = {len(self.clusters_)}, ({i}, {j}, {"%.2f" % edge.weight}) ...')

            # store the distance
            self.distances_[k - n_samples] = edge.weight

            self.parent_[i] = self.parent_[j] = k
            self.children_.append((i, j))
            # Keep track of the number of elements per cluster
            n_i = self.used_node_[i]  # number of observations
            n_j = self.used_node_[j]  # number of observations
            self.used_node_[k] = n_i + n_j
            self.used_node_[i] = self.used_node_[j] = False

            coord_col = join_func(A[i], A[j], self.used_node_, n_i,
                                  n_j)

            # transfer the detection id to global bbox id
            self.clusters_[k] = self.clusters_[i] + self.clusters_[j]
            heapify(self.clusters_[k])

            for l, d in coord_col:
                A[l].append(k, d)
                # Here we use the information from coord_col (containing the
                # distances) to update the heap
                # Using a heap to insert items at the correct place in a priority queue:
                heappush(inertia, _hierarchical.WeightedEdge(d, k,
                                                             l))
            A[k] = coord_col

            # # Clear A[i] and A[j] to save memory
            A[i] = A[j] = 0

            # Clear clusters[i] and clusters[j] to save memory clusters[i] = clusters[j] = 0
            # remove the key, value from the dict
            del self.clusters_[i], self.clusters_[j]

    def _get_n_nodes(self):
        return 2 * self.n_samples_ - 1

    def _init_A(self):
        n_nodes = self._get_n_nodes()
        A = np.empty(n_nodes, dtype=object)
        return A

    def _generate_cannot_link_mask(self, global_bbox_id_list):
        # print(f'Constructing can not link for detections in the same image ... ')
        # start_time = time.time()
        # location of 1 means can link, 0 means cannot link
        ahc_tool = AHC()
        ahc_tool.video_detected_bbox_all = self.video_detected_bbox_all
        mask = ahc_tool.estimate_same_image_mask(global_bbox_id_list)
        # print("Constructing can not link for detections in the same image done in "
        #       "[--- %s seconds ---]" % (time.time() - start_time))
        return mask

    # def _impose_cannot_link(self, cannot_link_mask, cannot_link_global_bbox_id_list):
    #     pass

    def _extend_same_cluster_det(self, global_bbox_id_list, include_same_cluster_det):
        node_ids = self._hc_get_unique_heads(global_bbox_id_list)

        clusters_related = {}
        global_bbox_id_list_new = []
        for n_k in node_ids:
            same_cluster_global_bbox_id_list = self.heap_cluster_global_bbox_id_list(self.clusters_[n_k])
            if include_same_cluster_det:
                clusters_related[n_k] = same_cluster_global_bbox_id_list
                global_bbox_id_list_new.extend(same_cluster_global_bbox_id_list)
            else:
                # filter out the det that is not in the current sub-sequence
                clusters_related[n_k] = [x for x in same_cluster_global_bbox_id_list if x in global_bbox_id_list]

        # replace the global_bbox_id_list if include_same_cluster_det
        if include_same_cluster_det:
            global_bbox_id_list = global_bbox_id_list_new

        return global_bbox_id_list, clusters_related

    def _remove_det_list_for_clusters_of_given_length(self, global_bbox_id_list, num_frame_per_sub_seq):
        global_bbox_id_list_refined = global_bbox_id_list.copy()
        _, clusters_related = self._extend_same_cluster_det(
            global_bbox_id_list_refined, include_same_cluster_det=False
        )
        related_nodes = []
        for n_k, cluster in clusters_related.items():
            if len(cluster) == num_frame_per_sub_seq:
                [global_bbox_id_list_refined.remove(x) for x in cluster]
                related_nodes.append(n_k)
        return global_bbox_id_list_refined, related_nodes

    def _init_inertia_from_pdist_by_mask(self, global_bbox_id_list, pdist=None, mask=None, **kwargs):
        A = self._init_A()
        inertia = list()
        # print(f'Function _init_inertia_from_pdist_by_mask: '
        #       f'len(global_bbox_id_list) = {len(global_bbox_id_list)} ...')

        # build the correspondence for sample id in pdist and global bbox id
        DetID2GlobalBboxID_Array, GlobalBboxID2DetID_Dict = self.build_reference(global_bbox_id_list)

        if pdist is None:
            pdist = self._computer_pdist(global_bbox_id_list, with_st_const=False)
        if mask is None:
            mask = self._generate_cannot_link_mask(global_bbox_id_list)

        # start_time = time.time()
        for i in range(len(global_bbox_id_list)):
            pdist_row = np.where(mask[i, :] != 0)[0]  # np.where return a tuple
            n_l_array = DetID2GlobalBboxID_Array[pdist_row]
            data = pdist[i, pdist_row]

            n_k = DetID2GlobalBboxID_Array[i]
            A[n_k] = IntFloatDict(np.array(n_l_array, dtype=np.intp),
                                  np.array(data, dtype=np.float64))  # fast dict
            # We keep only the upper triangular for the heap
            #  Generator expressions are faster than arrays on the following
            # WeightedEdge(weight=0.021998, a=0, b=53)
            inertia.extend(_hierarchical.WeightedEdge(d, n_k, n_l)
                           for n_l, d in zip(n_l_array, data) if n_l > n_k)

        # heapify(inertia)  # let the later function to conduct this.
        return inertia, A

    # def _init_inertia_from_pdist_for_clusters(self, global_bbox_id_list, A=None, inertia=None, **kwargs):
    #     if A is None:
    #         A = self._init_A()
    #
    #     if inertia is None:
    #         inertia = list()
    #
    #     start_time = time.time()
    #     node_ids = self._hc_get_unique_heads(global_bbox_id_list)
    #     num_clusters = len(node_ids)
    #
    #     pdist = self._computer_pdist(global_bbox_id_list, with_st_const=False)
    #     # build the correspondence for sample id in pdist and global bbox id
    #     DetID2GlobalBboxID_Array, GlobalBboxID2DetID_Dict = self.build_reference(global_bbox_id_list)
    #     mask = self._generate_cannot_link_mask(global_bbox_id_list)
    #
    #     cluster_pdist_dict = {x: [] for x in node_ids}  # 2d array, first [] saves the rows, second the data.
    #     for k in range(num_clusters - 1):
    #         n_k = node_ids[k]
    #         # global bbox id list can be obtained by either self.clusters_[n_k] or
    #         # self.get_leaves(children=self.children_, n_leaves=self.n_samples_, node_id=n_k)
    #         # rows = [GlobalBboxID2DetID_Dict[x] for x in
    #         #         self.heap_cluster_global_bbox_id_list(self.clusters_[n_k])]
    #         rows = [GlobalBboxID2DetID_Dict[x] for x in clusters_related[n_k]]
    #
    #         # find the columns that can not merge with current n_k
    #         # can_merge_pdist_row = np.where(np.sum(mask[rows, :] != 0, axis=0) == len(rows))
    #         # It is wrong to find can merge pdist row
    #         cannot_merge_pdist_row = np.where(np.sum(mask[rows, :] != 0, axis=0) != len(rows))
    #         n_l_global_id_list = DetID2GlobalBboxID_Array[cannot_merge_pdist_row]
    #         cannot_link_nodes = self._hc_get_unique_heads(n_l_global_id_list)
    #         # data = pdist[i, pdist_row]
    #
    #         for l in range(k + 1, num_clusters):
    #             n_l = node_ids[l]
    #             if n_l in cannot_link_nodes: continue
    #
    #             # We impose spatial-temporal constraints for the global_bbox_id_list to link.
    #             # If two clusters have overlap in image id (e.g., two detections may be in the same image)
    #             # Here we always check if the whole cluster for the related global bbox id to link.
    #             image_ids = self.heap_cluster_image_id_list(self.clusters_[n_k] + self.clusters_[n_l])
    #             # image_ids = self.get_image_id_list_for_cluster(clusters_related[n_k] + clusters_related[n_l])
    #
    #             # The following operation should also works.
    #             # image_ids = self.heap_cluster_image_id_list(clusters_related[n_k] + clusters_related[n_l])
    #             # print(f'image_ids={image_ids}, n_k={n_k}, self.clusters_[n_k] = {self.clusters_[n_k]},'
    #             #       f'n_l={n_l}, self.clusters_[n_l]={self.clusters_[n_l]}')
    #             assert self.is_unique_list(image_ids)
    #
    #             # cols = [GlobalBboxID2DetID_Dict[x] for x in
    #             #         self.heap_cluster_global_bbox_id_list(self.clusters_[n_l])]
    #             cols = [GlobalBboxID2DetID_Dict[x] for x in clusters_related[n_l]]
    #             dist = AhcMetric.cluster_dist_from_pdist_rows_cols(
    #                 pdist=pdist, rows=rows, cols=cols, linkage=linkage
    #             )
    #
    #             cluster_pdist_dict[n_k].append([n_l, dist])
    #             cluster_pdist_dict[n_l].append([n_k, dist])
    #
    #             if n_l < n_k:
    #                 # heappush(inertia, _hierarchical.WeightedEdge(dist, n_l, n_k))
    #                 inertia.append(_hierarchical.WeightedEdge(dist, n_l, n_k))
    #             else:
    #                 # heappush(inertia, _hierarchical.WeightedEdge(dist, n_k, n_l))
    #                 inertia.append(_hierarchical.WeightedEdge(dist, n_k, n_l))
    #     heapify(inertia)

    def _init_inertia_from_pdist_by_cluster(
            self, clusters_related, global_bbox_id_list, linkage,
            pdist=None, mask=None
    ):
        node_ids = list(clusters_related.keys())
        num_clusters = len(node_ids)
        DetID2GlobalBboxID_Array, GlobalBboxID2DetID_Dict = self.build_reference(global_bbox_id_list)

        constraint_specified = True if pdist is not None or mask is not None else False
        if pdist is None:  # if we do not specify any constraints by pdist
            pdist = self._computer_pdist(global_bbox_id_list, with_st_const=False)
        if mask is None:  # if we do not specify any constraints by mask
            mask = self._generate_cannot_link_mask(global_bbox_id_list)

        single_det_nodes = [x for x in clusters_related.keys() if len(clusters_related[x]) == 1]  # e.g., 894 nodes
        multi_det_nodes = [x for x in clusters_related.keys() if len(clusters_related[x]) > 1]  # e.g., 129 nodes
        single_det_cluster_global_bbox_list = [
            clusters_related[x][0] for x in single_det_nodes
        ]
        inertia, A = list(), self._init_A()
        if len(single_det_nodes) > 1:  # if there is more than one single_det_nodes
            # update A or not (if all the single_det_cluster_global_bbox_list are in the same frame) in this branch
            if not constraint_specified:
                inertia, A = self._init_inertia_from_pdist_by_mask(
                    single_det_cluster_global_bbox_list
                    # extract the global bbox id list for the single det clusters
                )
            else:
                related_rows = np.array([GlobalBboxID2DetID_Dict[x] for x in single_det_cluster_global_bbox_list])
                inertia, A = self._init_inertia_from_pdist_by_mask(
                    single_det_cluster_global_bbox_list,
                    pdist=pdist[np.ix_(related_rows, related_rows)],
                    mask=mask[np.ix_(related_rows, related_rows)]
                    # extract the global bbox id list for the single det clusters
                )

        # padding (-1,-1) for the empty node of node_ids in A
        padding_nodes = [x for x in node_ids if A[x] is None or len(A[x]) == 0]
        fill_in_empty_fast_dict(A, padding_nodes)

        # # build the correspondence for sample id in pdist and global bbox id
        # DetID2GlobalBboxID_Array, GlobalBboxID2DetID_Dict = self.build_reference(global_bbox_id_list)
        # if pdist is None:  # if we do not specify any constraints by pdist
        #     pdist = self._computer_pdist(global_bbox_id_list, with_st_const=False)
        # if mask is None:  # if we do not specify any constraints by mask
        #     mask = self._generate_cannot_link_mask(global_bbox_id_list)

        # cluster_pdist_dict = {x: [] for x in node_ids}  # 2d array, first [] saves the rows, second the data.
        for k in range(num_clusters - 1):
            n_k = node_ids[k]
            # global bbox id list can be obtained by either self.clusters_[n_k] or
            # self.get_leaves(children=self.children_, n_leaves=self.n_samples_, node_id=n_k)
            # rows = [GlobalBboxID2DetID_Dict[x] for x in
            #         self.heap_cluster_global_bbox_id_list(self.clusters_[n_k])]
            rows = [GlobalBboxID2DetID_Dict[x] for x in clusters_related[n_k]]

            # find the columns that can not merge with current n_k
            # can_merge_pdist_row = np.where(np.sum(mask[rows, :] != 0, axis=0) == len(rows))
            # It is wrong to find can merge pdist row
            cannot_merge_pdist_row = np.where(np.sum(mask[rows, :] != 0, axis=0) != len(rows))
            n_l_global_id_list = DetID2GlobalBboxID_Array[cannot_merge_pdist_row]
            cannot_link_nodes = self._hc_get_unique_heads(n_l_global_id_list)
            # data = pdist[i, pdist_row]

            for l in range(k + 1, num_clusters):
                n_l = node_ids[l]
                if n_l in cannot_link_nodes or \
                        (len(clusters_related[n_l]) == 1 and
                         len(clusters_related[n_k]) == 1):  # single det clusters have been handled above
                    continue

                # # We impose spatial-temporal constraints for the global_bbox_id_list to link.
                # # If two clusters have overlap in image id (e.g., two detections may be in the same image)
                # # Here we always check if the whole cluster for the related global bbox id to link.
                # image_ids = self.heap_cluster_image_id_list(self.clusters_[n_k] + self.clusters_[n_l])
                # # image_ids = self.get_image_id_list_for_cluster(clusters_related[n_k] + clusters_related[n_l])
                #
                # # The following operation should also works.
                # # image_ids = self.heap_cluster_image_id_list(clusters_related[n_k] + clusters_related[n_l])
                # # print(f'image_ids={image_ids}, n_k={n_k}, self.clusters_[n_k] = {self.clusters_[n_k]},'
                # #       f'n_l={n_l}, self.clusters_[n_l]={self.clusters_[n_l]}')
                # assert self.is_unique_list(image_ids)

                # cols = [GlobalBboxID2DetID_Dict[x] for x in
                #         self.heap_cluster_global_bbox_id_list(self.clusters_[n_l])]
                cols = [GlobalBboxID2DetID_Dict[x] for x in clusters_related[n_l]]
                dist = AhcMetric.cluster_dist_from_pdist_rows_cols(
                    pdist=pdist, rows=rows, cols=cols, linkage=linkage
                )

                # cluster_pdist_dict[n_k].append([n_l, dist])
                # cluster_pdist_dict[n_l].append([n_k, dist])

                # if n_k == 7977 and n_l == 2682 and len(A[n_l]) == 0:
                #     print(f'Stop')
                # # print(
                # #     f'n_k = {n_k}, n_l={n_l} ')
                #     # f', A[n_k] = {A[n_k].to_arrays()[0][0]}, A[n_l] = {A[n_l].to_arrays()[0][0]}'
                A[n_k].append(n_l, dist)  # A[l].append(k, d)
                A[n_l].append(n_k, dist)
                # the following is slow
                # if n_k in A:
                #     A[n_k].append(n_l, dist)  # A[l].append(k, d)
                # else:
                #     A[n_k] = IntFloatDict(np.array([n_l], dtype=np.intp), np.array([dist], dtype=np.float64))
                #
                # if n_l in A:
                #     A[n_l].append(n_k, dist)
                # else:
                #     A[n_l] = IntFloatDict(np.array([n_k], dtype=np.intp), np.array([dist], dtype=np.float64))

                if n_l < n_k:
                    # heappush(inertia, _hierarchical.WeightedEdge(dist, n_l, n_k))
                    inertia.append(_hierarchical.WeightedEdge(dist, n_l, n_k))
                else:
                    # heappush(inertia, _hierarchical.WeightedEdge(dist, n_k, n_l))
                    inertia.append(_hierarchical.WeightedEdge(dist, n_k, n_l))

        fast_dict_remove_fill_in(A, padding_nodes)
        # assert that all the fast_dict is valid
        if len(inertia) > 0:
            for fast_dict in A:
                if fast_dict is not None and len(fast_dict.to_arrays()[0]) > 0:
                    assert fast_dict.to_arrays()[0][0] != -1

        return inertia, A

    @staticmethod
    def assert_inertia_A_equal(inertia1_A1, inertia2_A2):
        """
        Check the consistence of 'inertia' and 'A' of two methods.
        :param inertia1_A1:
        :param inertia2_A2:
        :return:
        """
        inertia1, A1 = inertia1_A1
        inertia2, A2 = inertia2_A2

        assert len(inertia1) == len(inertia2)
        assert len(A1) == len(A2)

        # we must sort the edges to compare the sequence
        heapify(inertia1)
        heapify(inertia2)

        while inertia1:
            edge1 = heappop(inertia1)
            edge = heappop(inertia2)
            assert edge == edge1

        for k in range(len(A2)):
            if A2[k] is None:
                assert A1[k] is None
            else:
                assert len(A2[k]) == len(A1[k])
        print(f' [assert_inertia_A_equal()] Passed')

    def _init_inertia_by_pdist(
            self, global_bbox_id_list, linkage,
            include_same_cluster_det=INCLUDE_SAME_CLUSTER_DET,
            **kwargs
    ):
        """
        Speed test:
            inertia = []
            heappush(inertia, _hierarchical.WeightedEdge(dist, n_l, n_k))
        exhibited almost the same speed as
            inertia = list()
            inertia.append(_hierarchical.WeightedEdge(dist, n_l, n_k))
            heapify(inertia),
        0.76 vs 0.7519
        0.067 vs 0.066
        0.0697 vs 0.06329
        0.061 vs 0.0608
        because the former has to conduct insert sort.
        :param global_bbox_id_list:
        :param linkage:
        :param include_same_cluster_det:
        :param kwargs:
        :return:
        """

        assert len(global_bbox_id_list) > 0,\
            f'At least one global bbox id should be given for initializing inertia ...'

        node_ids = self._hc_get_unique_heads(global_bbox_id_list)
        num_clusters = len(node_ids)  # for this batch
        if num_clusters == 1:
            print(f'num_clusters = {num_clusters}, return empty inertia ...')
            # If there is only a single cluster, then no future merging will occur.
            return [], None  # set inertia to empty list, A to None

        global_bbox_id_list, clusters_related = self._extend_same_cluster_det(
            global_bbox_id_list=global_bbox_id_list,
            include_same_cluster_det=include_same_cluster_det
        )

        if len(global_bbox_id_list) == num_clusters:
            inertia, A = self._init_inertia_from_pdist_by_mask(global_bbox_id_list)
        else:
            inertia, A = self._init_inertia_from_pdist_by_cluster(
                clusters_related=clusters_related,
                global_bbox_id_list=global_bbox_id_list,
                linkage=linkage)

            ########################################
            # Second version
            ########################################

            # single_det_nodes = [x for x in clusters_related.keys() if len(clusters_related[x]) == 1]  # e.g., 894 nodes
            # multi_det_nodes = [x for x in clusters_related.keys() if len(clusters_related[x]) > 1]  # e.g., 129 nodes
            # single_det_cluster_global_bbox_list = [
            #     clusters_related[x][0] for x in single_det_nodes
            # ]
            # if len(single_det_nodes) > 1:  # if there is more than one single_det_nodes
            #     inertia, A = self._init_inertia_from_pdist_by_mask(
            #         single_det_cluster_global_bbox_list  # extract the global bbox id list for the single det clusters
            #     )
            # else:
            #     inertia, A = list(), self._init_A()
            #     if len(single_det_nodes) == 1:
            #         fill_in_empty_fast_dict(A, single_det_nodes)
            #
            # if len(multi_det_nodes) > 0:
            #     fill_in_empty_fast_dict(A, multi_det_nodes)
            #
            #     pdist = self._computer_pdist(global_bbox_id_list, with_st_const=False)
            #     # build the correspondence for sample id in pdist and global bbox id
            #     DetID2GlobalBboxID_Array, GlobalBboxID2DetID_Dict = self.build_reference(global_bbox_id_list)
            #     mask = self._generate_cannot_link_mask(global_bbox_id_list)
            #
            #     # cluster_pdist_dict = {x: [] for x in node_ids}  # 2d array, first [] saves the rows, second the data.
            #     for k in range(num_clusters - 1):
            #         n_k = node_ids[k]
            #         # global bbox id list can be obtained by either self.clusters_[n_k] or
            #         # self.get_leaves(children=self.children_, n_leaves=self.n_samples_, node_id=n_k)
            #         # rows = [GlobalBboxID2DetID_Dict[x] for x in
            #         #         self.heap_cluster_global_bbox_id_list(self.clusters_[n_k])]
            #         rows = [GlobalBboxID2DetID_Dict[x] for x in clusters_related[n_k]]
            #
            #         # find the columns that can not merge with current n_k
            #         # can_merge_pdist_row = np.where(np.sum(mask[rows, :] != 0, axis=0) == len(rows))
            #         # It is wrong to find can merge pdist row
            #         cannot_merge_pdist_row = np.where(np.sum(mask[rows, :] != 0, axis=0) != len(rows))
            #         n_l_global_id_list = DetID2GlobalBboxID_Array[cannot_merge_pdist_row]
            #         cannot_link_nodes = self._hc_get_unique_heads(n_l_global_id_list)
            #         # data = pdist[i, pdist_row]
            #
            #         for l in range(k + 1, num_clusters):
            #             n_l = node_ids[l]
            #             if n_l in cannot_link_nodes or \
            #                     (len(clusters_related[n_l]) == 1 and
            #                      len(clusters_related[n_k]) == 1):  # single det clusters have been handled above
            #                 continue
            #
            #             # # We impose spatial-temporal constraints for the global_bbox_id_list to link.
            #             # # If two clusters have overlap in image id (e.g., two detections may be in the same image)
            #             # # Here we always check if the whole cluster for the related global bbox id to link.
            #             # image_ids = self.heap_cluster_image_id_list(self.clusters_[n_k] + self.clusters_[n_l])
            #             # # image_ids = self.get_image_id_list_for_cluster(clusters_related[n_k] + clusters_related[n_l])
            #             #
            #             # # The following operation should also works.
            #             # # image_ids = self.heap_cluster_image_id_list(clusters_related[n_k] + clusters_related[n_l])
            #             # # print(f'image_ids={image_ids}, n_k={n_k}, self.clusters_[n_k] = {self.clusters_[n_k]},'
            #             # #       f'n_l={n_l}, self.clusters_[n_l]={self.clusters_[n_l]}')
            #             # assert self.is_unique_list(image_ids)
            #
            #             # cols = [GlobalBboxID2DetID_Dict[x] for x in
            #             #         self.heap_cluster_global_bbox_id_list(self.clusters_[n_l])]
            #             cols = [GlobalBboxID2DetID_Dict[x] for x in clusters_related[n_l]]
            #             dist = AhcMetric.cluster_dist_from_pdist_rows_cols(
            #                 pdist=pdist, rows=rows, cols=cols, linkage=linkage
            #             )
            #
            #             # cluster_pdist_dict[n_k].append([n_l, dist])
            #             # cluster_pdist_dict[n_l].append([n_k, dist])
            #
            #             A[n_k].append(n_l, dist)  # A[l].append(k, d)
            #             A[n_l].append(n_k, dist)
            #             # the following is slow
            #             # if n_k in A:
            #             #     A[n_k].append(n_l, dist)  # A[l].append(k, d)
            #             # else:
            #             #     A[n_k] = IntFloatDict(np.array([n_l], dtype=np.intp), np.array([dist], dtype=np.float64))
            #             #
            #             # if n_l in A:
            #             #     A[n_l].append(n_k, dist)
            #             # else:
            #             #     A[n_l] = IntFloatDict(np.array([n_k], dtype=np.intp), np.array([dist], dtype=np.float64))
            #
            #             if n_l < n_k:
            #                 # heappush(inertia, _hierarchical.WeightedEdge(dist, n_l, n_k))
            #                 inertia.append(_hierarchical.WeightedEdge(dist, n_l, n_k))
            #             else:
            #                 # heappush(inertia, _hierarchical.WeightedEdge(dist, n_k, n_l))
            #                 inertia.append(_hierarchical.WeightedEdge(dist, n_k, n_l))
            #
            #     # remove the (-1, -1) element for each IntFloatDict
            #     fast_dict_remove_fill_in(A, multi_det_nodes)
            #
            # if len(single_det_nodes) == 1:
            #     fast_dict_remove_fill_in(A, single_det_nodes)
            # # assert that all the fast_dict is valid
            # for fast_dict in A:
            #     if fast_dict is not None:
            #         assert fast_dict.to_arrays()[0][0] != -1

            # self.assert_inertia_A_equal(inertia1_A1=(inertia1, A1), inertia2_A2=(inertia, A))
            # assert len(inertia1) == len(inertia)
            # while inertia1:
            #     edge1 = heappop(inertia1)
            #     edge = heappop(inertia)
            #     assert edge == edge1
            # assert len(A1) == len(A)
            # for k in range(len(A)):
            #     if A[k] is None:
            #         assert A1[k] is None
            #     else:
            #         assert len(A[k]) == len(A1[k])
            #
            # print(f'Passed verified')
            ####################################
            # slow version, deprecated.
            ###########################
            # cluster_pdist_dict = {x: [] for x in node_ids}  # 2d array, first [] saves the rows, second the data.
            # for k in range(num_clusters - 1):
            #     n_k = node_ids[k]
            #     # global bbox id list can be obtained by either self.clusters_[n_k] or
            #     # self.get_leaves(children=self.children_, n_leaves=self.n_samples_, node_id=n_k)
            #     # rows = [GlobalBboxID2DetID_Dict[x] for x in
            #     #         self.heap_cluster_global_bbox_id_list(self.clusters_[n_k])]
            #     rows = [GlobalBboxID2DetID_Dict[x] for x in clusters_related[n_k]]
            #
            #     for l in range(k + 1, num_clusters):
            #         n_l = node_ids[l]
            #
            #         # We impose spatial-temporal constraints for the global_bbox_id_list to link.
            #         # If two clusters have overlap in image id (e.g., two detections may be in the same image)
            #         # Here we always check if the whole cluster for the related global bbox id to link.
            #         image_ids = self.heap_cluster_image_id_list(self.clusters_[n_k] + self.clusters_[n_l])
            #
            #         # The following operation should also works.
            #         # image_ids = self.heap_cluster_image_id_list(clusters_related[n_k] + clusters_related[n_l])
            #
            #         if not self.is_unique_list(image_ids):
            #             continue
            #
            #         # cols = [GlobalBboxID2DetID_Dict[x] for x in
            #         #         self.heap_cluster_global_bbox_id_list(self.clusters_[n_l])]
            #         cols = [GlobalBboxID2DetID_Dict[x] for x in clusters_related[n_l]]
            #         dist = AhcMetric.cluster_dist_from_pdist_rows_cols(pdist=pdist, rows=rows, cols=cols,
            #             linkage=linkage
            #         )
            #
            #         cluster_pdist_dict[n_k].append([n_l, dist])
            #         cluster_pdist_dict[n_l].append([n_k, dist])
            #
            #         if n_l < n_k:
            #             # heappush(inertia, _hierarchical.WeightedEdge(dist, n_l, n_k))
            #             inertia.append(_hierarchical.WeightedEdge(dist, n_l, n_k))
            #         else:
            #             # heappush(inertia, _hierarchical.WeightedEdge(dist, n_k, n_l))
            #             inertia.append(_hierarchical.WeightedEdge(dist, n_k, n_l))
            # heapify(inertia)

            # A = self._cluster_dist_to_edge(cluster_pdist_dict)

        heapify(inertia)
        # print(f'Initializing A and inertia done in {time.time() - start_time} seconds. '
        #       f'len(inertia) = {len(inertia)} ... ')

        return inertia, A

    def _init_inertia_by_pdist_for_restored_dets_slow(
            self, global_bbox_id_list, restored_det_global_bbox_id_list,
            include_same_cluster_det=INCLUDE_SAME_CLUSTER_DET,
            **kwargs
    ):
        """
        update the pdist so that "the detections in existing clusters" [Group 1 detections] has
        large dist (they are valid to be connected, but the dist is so large so that they will not
        be considered as the same cluster during the merging), and the restored detections
        [Group 2 detections] has large dist with each other, but the distance between the
        detections in existing clusters and the restored detections remains untouched.
        Only single linkage is valid for initializing inertia, as modifying the dist to LARGE VALUE will
        completely disable the merging of [Group 1 detections], or merging of [Group 2 detections] if 'average'
        or 'complete' linkage is used.
        :param global_bbox_id_list:
        :param restored_det_global_bbox_id_list:
        :param include_same_cluster_det:
        :param kwargs:
        :return:
        """
        start_time = time.time()
        # check all the restored det are included in global_bbox_id_list
        for x in restored_det_global_bbox_id_list:
            assert x in global_bbox_id_list

        assert len(global_bbox_id_list) > 0 and len(restored_det_global_bbox_id_list) > 0,\
            f'At least one global bbox id, and one restored det should be given for initializing inertia ...'
        assert 'linkage' not in kwargs, f'Function _init_inertia_by_pdist_for_restored_dets only handle ' \
                                        f'"single" linkage, but linkage {kwargs["linkage"]} was given ...'
        node_ids = self._hc_get_unique_heads(global_bbox_id_list)
        num_clusters = len(node_ids)  # for this batch
        if num_clusters == 1:
            print(f'num_clusters = {num_clusters}, return empty inertia ...')
            # If there is only a single cluster, then no future merging will occur.
            return [], None  # set inertia to empty list, A to None

        global_bbox_id_list, clusters_related = self._extend_same_cluster_det(
            global_bbox_id_list=global_bbox_id_list,
            include_same_cluster_det=include_same_cluster_det
        )

        inertia, A = list(), self._init_A()
        fill_in_empty_fast_dict(A, node_ids)

        pdist = self._computer_pdist(global_bbox_id_list, with_st_const=False)
        # build the correspondence for sample id in pdist and global bbox id
        DetID2GlobalBboxID_Array, GlobalBboxID2DetID_Dict = self.build_reference(global_bbox_id_list)
        mask = self._generate_cannot_link_mask(global_bbox_id_list)

        # update pdist
        existing_clusters_det_rows = np.array(
            [GlobalBboxID2DetID_Dict[x] for x in global_bbox_id_list
             if x not in restored_det_global_bbox_id_list])
        pdist[np.ix_(existing_clusters_det_rows, existing_clusters_det_rows)] = CANNOT_LINK_DIST

        false_positive_det_rows = np.array([GlobalBboxID2DetID_Dict[x]
                                            for x in restored_det_global_bbox_id_list])
        pdist[np.ix_(false_positive_det_rows, false_positive_det_rows)] = CANNOT_LINK_DIST

        # cluster_pdist_dict = {x: [] for x in node_ids}  # 2d array, first [] saves the rows, second the data.
        for k in range(num_clusters - 1):
            n_k = node_ids[k]
            # global bbox id list can be obtained by either self.clusters_[n_k] or
            # self.get_leaves(children=self.children_, n_leaves=self.n_samples_, node_id=n_k)
            # rows = [GlobalBboxID2DetID_Dict[x] for x in
            #         self.heap_cluster_global_bbox_id_list(self.clusters_[n_k])]
            rows = [GlobalBboxID2DetID_Dict[x] for x in clusters_related[n_k]]

            # find the columns that can not merge with current n_k
            # can_merge_pdist_row = np.where(np.sum(mask[rows, :] != 0, axis=0) == len(rows))
            # It is wrong to find can merge pdist row
            cannot_merge_pdist_row = np.where(np.sum(mask[rows, :] != 0, axis=0) != len(rows))
            n_l_global_id_list = DetID2GlobalBboxID_Array[cannot_merge_pdist_row]
            cannot_link_nodes = self._hc_get_unique_heads(n_l_global_id_list)
            # data = pdist[i, pdist_row]

            for l in range(k + 1, num_clusters):
                n_l = node_ids[l]
                if n_l in cannot_link_nodes:
                    continue

                # We impose spatial-temporal constraints for the global_bbox_id_list to link.
                # If two clusters have overlap in image id (e.g., two detections may be in the same image)
                # Here we always check if the whole cluster for the related global bbox id to link.
                image_ids = self.heap_cluster_image_id_list(self.clusters_[n_k] + self.clusters_[n_l])
                # image_ids = self.get_image_id_list_for_cluster(clusters_related[n_k] + clusters_related[n_l])

                # The following operation should also works.
                # image_ids = self.heap_cluster_image_id_list(clusters_related[n_k] + clusters_related[n_l])
                # print(f'image_ids={image_ids}, n_k={n_k}, self.clusters_[n_k] = {self.clusters_[n_k]},'
                #       f'n_l={n_l}, self.clusters_[n_l]={self.clusters_[n_l]}')
                assert self.is_unique_list(image_ids)

                # cols = [GlobalBboxID2DetID_Dict[x] for x in
                #         self.heap_cluster_global_bbox_id_list(self.clusters_[n_l])]
                cols = [GlobalBboxID2DetID_Dict[x] for x in clusters_related[n_l]]
                dist = AhcMetric.cluster_dist_from_pdist_rows_cols(
                    pdist=pdist, rows=rows, cols=cols, linkage='single'
                )

                # cluster_pdist_dict[n_k].append([n_l, dist])
                # cluster_pdist_dict[n_l].append([n_k, dist])

                A[n_k].append(n_l, dist)  # A[l].append(k, d)
                A[n_l].append(n_k, dist)
                # the following is slow
                # if n_k in A:
                #     A[n_k].append(n_l, dist)  # A[l].append(k, d)
                # else:
                #     A[n_k] = IntFloatDict(np.array([n_l], dtype=np.intp), np.array([dist], dtype=np.float64))
                #
                # if n_l in A:
                #     A[n_l].append(n_k, dist)
                # else:
                #     A[n_l] = IntFloatDict(np.array([n_k], dtype=np.intp), np.array([dist], dtype=np.float64))

                if n_l < n_k:
                    # heappush(inertia, _hierarchical.WeightedEdge(dist, n_l, n_k))
                    inertia.append(_hierarchical.WeightedEdge(dist, n_l, n_k))
                else:
                    # heappush(inertia, _hierarchical.WeightedEdge(dist, n_k, n_l))
                    inertia.append(_hierarchical.WeightedEdge(dist, n_k, n_l))

        fast_dict_remove_fill_in(A, node_ids)
        # assert that all the fast_dict is valid
        for fast_dict in A:
            if fast_dict is not None:
                assert fast_dict.to_arrays()[0][0] != -1

        heapify(inertia)
        print(f'Initializing A and inertia done in {time.time() - start_time} seconds. '
              f'len(inertia) = {len(inertia)} ... ')

        return inertia, A

    def _init_inertia_by_pdist_for_restored_dets(
            self, global_bbox_id_list, restored_det_global_bbox_id_list,
            include_same_cluster_det=INCLUDE_SAME_CLUSTER_DET, seq_max_length=None,
            **kwargs
    ):
        """
        update the pdist so that "the detections in existing clusters" [Group 1 detections] has
        large dist (they are valid to be connected, but the dist is so large so that they will not
        be considered as the same cluster during the merging), and the restored detections
        [Group 2 detections] has large dist with each other, but the distance between the
        detections in existing clusters and the restored detections remains untouched.
        Only single linkage is valid for initializing inertia, as modifying the dist to LARGE VALUE will
        completely disable the merging of [Group 1 detections], or merging of [Group 2 detections] if 'average'
        or 'complete' linkage is used.
        :param global_bbox_id_list:
        :param restored_det_global_bbox_id_list:
        :param include_same_cluster_det:
        :param kwargs:
        :return:
        """
        LINKAGE = 'single'  # only 'single' linkage is available for this function

        start_time = time.time()
        # check all the restored det are included in global_bbox_id_list
        for x in restored_det_global_bbox_id_list:
            assert x in global_bbox_id_list

        assert len(global_bbox_id_list) > 0 and len(restored_det_global_bbox_id_list) > 0,\
            f'At least one global bbox id, and one restored det should be given for initializing inertia ...'
        assert 'linkage' not in kwargs, f'Function _init_inertia_by_pdist_for_restored_dets only handle ' \
                                        f'"single" linkage, but linkage {kwargs["linkage"]} was given ...'
        node_ids = self._hc_get_unique_heads(global_bbox_id_list)
        num_clusters = len(node_ids)  # for this batch
        if num_clusters == 1:
            print(f'num_clusters = {num_clusters}, return empty inertia ...')
            # If there is only a single cluster, then no future merging will occur.
            return [], None  # set inertia to empty list, A to None

        global_bbox_id_list, clusters_related = self._extend_same_cluster_det(
            global_bbox_id_list=global_bbox_id_list,
            include_same_cluster_det=include_same_cluster_det
        )

        if not include_same_cluster_det and seq_max_length is not None:
            if isinstance(seq_max_length, int) and seq_max_length > 0:
                # remove the global_bbox_id based on the length of the cluster
                global_bbox_id_list, related_nodes = self._remove_det_list_for_clusters_of_given_length(
                    global_bbox_id_list, seq_max_length
                )
                # Delete the corresponding keys from clusters_related so later we do not need consider it at all,
                # since there is no missed detections to fill for those clusters.
                for n_k in related_nodes:
                    del clusters_related[n_k]
                # print(f'{len(related_nodes) } full length nodes [len: {seq_max_length}] removed from clusters_related')

        inertia, A = list(), self._init_A()
        fill_in_empty_fast_dict(A, node_ids)

        pdist = self._computer_pdist(global_bbox_id_list, with_st_const=False)
        # build the correspondence for sample id in pdist and global bbox id
        DetID2GlobalBboxID_Array, GlobalBboxID2DetID_Dict = self.build_reference(global_bbox_id_list)
        mask = self._generate_cannot_link_mask(global_bbox_id_list)

        # update pdist
        # ------------------------- disable the connection for existing clusters -------------------------
        # existing_clusters_det_rows = np.array(
        #     [GlobalBboxID2DetID_Dict[x] for x in global_bbox_id_list
        #      if x not in restored_det_global_bbox_id_list])
        # pdist[np.ix_(existing_clusters_det_rows, existing_clusters_det_rows)] = CANNOT_LINK_DIST

        false_positive_det_rows = np.array([GlobalBboxID2DetID_Dict[x]
                                            for x in restored_det_global_bbox_id_list])
        pdist[np.ix_(false_positive_det_rows, false_positive_det_rows)] = CANNOT_LINK_DIST

        # inertia1, A1 =
        inertia, A = self._init_inertia_from_pdist_by_cluster(
            clusters_related=clusters_related,
            global_bbox_id_list=global_bbox_id_list,
            linkage=LINKAGE,
            pdist=pdist, mask=mask
        )

        # ------------------- deprecated version, test passed.
        # # cluster_pdist_dict = {x: [] for x in node_ids}  # 2d array, first [] saves the rows, second the data.
        # for k in range(num_clusters - 1):
        #     n_k = node_ids[k]
        #     # global bbox id list can be obtained by either self.clusters_[n_k] or
        #     # self.get_leaves(children=self.children_, n_leaves=self.n_samples_, node_id=n_k)
        #     # rows = [GlobalBboxID2DetID_Dict[x] for x in
        #     #         self.heap_cluster_global_bbox_id_list(self.clusters_[n_k])]
        #     rows = [GlobalBboxID2DetID_Dict[x] for x in clusters_related[n_k]]
        #
        #     # find the columns that can not merge with current n_k
        #     # can_merge_pdist_row = np.where(np.sum(mask[rows, :] != 0, axis=0) == len(rows))
        #     # It is wrong to find can merge pdist row
        #     cannot_merge_pdist_row = np.where(np.sum(mask[rows, :] != 0, axis=0) != len(rows))
        #     n_l_global_id_list = DetID2GlobalBboxID_Array[cannot_merge_pdist_row]
        #     cannot_link_nodes = self._hc_get_unique_heads(n_l_global_id_list)
        #     # data = pdist[i, pdist_row]
        #
        #     for l in range(k + 1, num_clusters):
        #         n_l = node_ids[l]
        #         if n_l in cannot_link_nodes:
        #             continue
        #
        #         # We impose spatial-temporal constraints for the global_bbox_id_list to link.
        #         # If two clusters have overlap in image id (e.g., two detections may be in the same image)
        #         # Here we always check if the whole cluster for the related global bbox id to link.
        #         image_ids = self.heap_cluster_image_id_list(self.clusters_[n_k] + self.clusters_[n_l])
        #         # image_ids = self.get_image_id_list_for_cluster(clusters_related[n_k] + clusters_related[n_l])
        #
        #         # The following operation should also works.
        #         # image_ids = self.heap_cluster_image_id_list(clusters_related[n_k] + clusters_related[n_l])
        #         # print(f'image_ids={image_ids}, n_k={n_k}, self.clusters_[n_k] = {self.clusters_[n_k]},'
        #         #       f'n_l={n_l}, self.clusters_[n_l]={self.clusters_[n_l]}')
        #         assert self.is_unique_list(image_ids)
        #
        #         # cols = [GlobalBboxID2DetID_Dict[x] for x in
        #         #         self.heap_cluster_global_bbox_id_list(self.clusters_[n_l])]
        #         cols = [GlobalBboxID2DetID_Dict[x] for x in clusters_related[n_l]]
        #         dist = AhcMetric.cluster_dist_from_pdist_rows_cols(
        #             pdist=pdist, rows=rows, cols=cols, linkage=LINKAGE
        #         )
        #
        #         # cluster_pdist_dict[n_k].append([n_l, dist])
        #         # cluster_pdist_dict[n_l].append([n_k, dist])
        #
        #         A[n_k].append(n_l, dist)  # A[l].append(k, d)
        #         A[n_l].append(n_k, dist)
        #         # the following is slow
        #         # if n_k in A:
        #         #     A[n_k].append(n_l, dist)  # A[l].append(k, d)
        #         # else:
        #         #     A[n_k] = IntFloatDict(np.array([n_l], dtype=np.intp), np.array([dist], dtype=np.float64))
        #         #
        #         # if n_l in A:
        #         #     A[n_l].append(n_k, dist)
        #         # else:
        #         #     A[n_l] = IntFloatDict(np.array([n_k], dtype=np.intp), np.array([dist], dtype=np.float64))
        #
        #         if n_l < n_k:
        #             # heappush(inertia, _hierarchical.WeightedEdge(dist, n_l, n_k))
        #             inertia.append(_hierarchical.WeightedEdge(dist, n_l, n_k))
        #         else:
        #             # heappush(inertia, _hierarchical.WeightedEdge(dist, n_k, n_l))
        #             inertia.append(_hierarchical.WeightedEdge(dist, n_k, n_l))
        #
        # fast_dict_remove_fill_in(A, node_ids)
        # # assert that all the fast_dict is valid
        # for fast_dict in A:
        #     if fast_dict is not None:
        #         assert fast_dict.to_arrays()[0][0] != -1

        # self.assert_inertia_A_equal(inertia1_A1=(inertia1.copy(), A1.copy()),
        #                             inertia2_A2=(inertia.copy(), A.copy()))

        heapify(inertia)
        print(f'Initializing A and inertia done in {time.time() - start_time} seconds. '
              f'len(inertia) = {len(inertia)} ... ')

        return inertia, A

    def _init_inertia_by_dist_metric(
            self, global_bbox_id_list, cluster_dist_metric, linkage, with_st_const=True, **kwargs
    ):
        inertia = list()
        node_ids = self._hc_get_unique_heads(global_bbox_id_list)
        num_clusters = len(node_ids)  # for this batch
        for n_k in node_ids:
            assert len(self.clusters_[n_k]) > 0

        cluster_pdist_dict = {x: [] for x in node_ids}  # 2d array, first [] saves the rows, second the data.
        for k in range(num_clusters - 1):
            n_k = node_ids[k]
            heap_cluster1 = self.clusters_[n_k]
            # ---------------------- debug
            rows = self.get_leaves(children=self.children_, n_leaves=self.n_samples_, node_id=n_k)
            assert len(rows) == len(heap_cluster1)
            # ----------------------
            for l in range(k + 1, num_clusters):
                n_l = node_ids[l]
                heap_cluster2 = self.clusters_[n_l]

                # -------------------------------- do we need to check image id? think in future
                if with_st_const:
                    combined_cluster = self.clusters_[n_k] + self.clusters_[n_l]
                    image_ids = [x[0] for x in combined_cluster]
                    # We let the later process to decide whether they should be merged or not
                    # if two clusters have overlap in image id (e.g., two detections may be in the same image)
                    if not self.is_unique_list(image_ids):
                        continue
                # # -------------------------------------------
                cols = self.get_leaves(children=self.children_, n_leaves=self.n_samples_, node_id=n_l)
                assert len(cols) == len(heap_cluster2)

                dist = cluster_dist_metric(
                    heap_cluster1, heap_cluster2, linkage, **kwargs
                )
                # if appearance_dist >= 2:  # 1.0212946 ok, [-1,1] for similarity, 0,2 for distance 1 - simialrity
                #     print(f'self.temporal_distance(clusters[n_k], clusters[n_l]) ='
                #           f' {self.temporal_distance(clusters[n_k], clusters[n_l])}')
                # assert dist <= 2.0

                cluster_pdist_dict[n_k].append([n_l, dist])
                cluster_pdist_dict[n_l].append([n_k, dist])
                # We keep only the upper triangular for the heap
                # Generator expressions are faster than arrays on the following
                if n_l < n_k:
                    inertia.append(_hierarchical.WeightedEdge(dist, n_l, n_k))
                else:
                    inertia.append(_hierarchical.WeightedEdge(dist, n_k, n_l))
        heapify(inertia)

        A = self._cluster_dist_to_edge(cluster_pdist_dict)

        return inertia, A

    def tracking_one_subsequent(
            self, global_bbox_id_list, distance_threshold, linkage,
            major_dist_func=None, filter_config=None, **kwargs
    ):
        """Conduct tracking for a given global_bbox_id_list.
        :param filter_config:
        :param major_dist_func:
        :param global_bbox_id_list:
        :param distance_threshold:
        :param linkage:
        :return:
        """

        # num_cluster = len(self._hc_get_unique_heads(global_bbox_id_list))
        # if len(global_bbox_id_list) == num_cluster:

        if major_dist_func is not None:
            inertia, A = self._init_inertia_by_dist_metric(
                global_bbox_id_list=global_bbox_id_list,
                cluster_dist_metric=major_dist_func,
                linkage=linkage, **kwargs
            )
        else:
            inertia, A = self._init_inertia_by_pdist(
                global_bbox_id_list=global_bbox_id_list, linkage=linkage,
            )

        # no need to conduct tracking if there is no candidate merging
        if len(inertia) == 0:
            return
        self._subsequence_tracking(
            inertia=inertia, A=A, distance_threshold=distance_threshold,
            linkage=linkage, filter_config=filter_config, **kwargs)

    def _divide_sequence(self, subsequence_config=None):
        assert subsequence_config is None or isinstance(subsequence_config, dict)

        if subsequence_config is None:  # num_frame_per_sub_seq max_det_num_per_sub_seq not in
            max_det_num_per_sub_seq = self.batch_det_num
            D = self.auto_divide_sequence(
                max_det_num_per_sub_seq=max_det_num_per_sub_seq
            )
        else:  # isinstance(subsequence_config, dict):
            assert 'num_frame_per_sub_seq' in subsequence_config or \
                'max_det_num_per_sub_seq' in subsequence_config

            if 'num_frame_per_sub_seq' in subsequence_config:
                num_frame_per_sub_seq = subsequence_config['num_frame_per_sub_seq']
                D = self.auto_divide_sequence(
                    num_frame_per_sub_seq=num_frame_per_sub_seq
                )
                # done = True
            else:  # 'max_det_num_per_sub_seq' in subsequence_config:
                max_det_num_per_sub_seq = subsequence_config['max_det_num_per_sub_seq']
                D = self.auto_divide_sequence(
                    max_det_num_per_sub_seq=max_det_num_per_sub_seq
                )
                # done = True

        return D

    def filling_hole_one_subsequent(
            self, global_bbox_id_list, restored_det_global_bbox_id_list,
            distance_threshold, linkage, seq_max_length=None,
            major_dist_func=None, filter_config=None, **kwargs
    ):
        """Conduct tracking for a given global_bbox_id_list.
        :param seq_max_length:
        :param restored_det_global_bbox_id_list:
        :param filter_config:
        :param major_dist_func:
        :param global_bbox_id_list:
        :param distance_threshold:
        :param linkage:
        :return:
        """

        # num_cluster = len(self._hc_get_unique_heads(global_bbox_id_list))
        # if len(global_bbox_id_list) == num_cluster:

        # if major_dist_func is not None:
        #     inertia, A = self._init_inertia_by_dist_metric(
        #         global_bbox_id_list=global_bbox_id_list,
        #         cluster_dist_metric=major_dist_func,
        #         linkage=linkage, **kwargs
        #     )
        # else:
        inertia, A = self._init_inertia_by_pdist_for_restored_dets(
            global_bbox_id_list=global_bbox_id_list,
            restored_det_global_bbox_id_list=restored_det_global_bbox_id_list,
            seq_max_length=seq_max_length
        )

        # no need to conduct tracking if there is no candidate merging
        if len(inertia) == 0:
            return
        self._subsequence_tracking(
            inertia=inertia, A=A, distance_threshold=distance_threshold,
            linkage=linkage, filter_config=filter_config, **kwargs)


