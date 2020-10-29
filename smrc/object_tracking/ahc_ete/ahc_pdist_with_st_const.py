###########################################
# The tracking are based on pdist for selected detections, not
# all the detections.
############################################

import time
from heapq import heapify, heappop, heappush

import numpy as np
from sklearn.cluster import _hierarchical_fast as _hierarchical
from sklearn.utils._fast_dict import IntFloatDict

from object_tracking.ahc_ete._ahc import AHCTracker, AhcMetric
from object_tracking.ahc_ete.standard_ahc import AHC


class AHCSubSequenceTracker(AHC, AHCTracker):
    """
    Our implementation of the AHC algorithm with the spatial-temporal constraint that detections in the same
    image should not be in the same cluster.
    All single, average, complete linkages (with spatial-temporal constraints built-in) are available.
    The adaptations include:
        1) my implementation of the "max merge", "average merge" and "single merge" with spatial-
        temporal constraints built in C code;
        I also implemented "max merge", "average merge" in python (test are passed), but the C code is much fast than
        the python code, so the C code is used in default.
        2) allowing tracking using one linkage in the first phase and another linkage in the next phase from the
        ending point of previous phase.
    Note: This class only support a single similarity measure and all the tracking processes are based
    on the same self.pdist_, which is initialized in self._init_inertia_from_scratch().
    If the the later expert is used to continue the tracking with the progress made by the previous expert, then
    self.pdist_ will be rebuilt using the existing clusters with the setting of new linkage.

    The AHC algorithm without the spatial-temporal constraint is not implemented, as we can
    use the standard ahc algorithm to achieve this.
    """
    def __init__(self):
        super().__init__()

        self.local_used_node_ = None
        self.local_parent_ = None
        self.local_children_ = None
        self.local_distances_ = None
        self.local_clusters_ = None
        self.local_pdist_ = None

    def _is_valid_merging(self, heap_cluster1, heap_cluster2, **kwargs):
        """Check if two merging are valid based on additional constraints, return true in default
        (i.e., no further checking). Any customized constraints (kf dist, etc.) should be implemented in subclasses.
        :param heap_cluster1:
        :param heap_cluster2:
        :return:
        """
        # return True or False
        return True

    def _init_inertia_from_scratch(self, global_bbox_id_list):
        pdist = self._computer_pdist(global_bbox_id_list, with_st_const=False)

        # Return a new array of given shape and type, without initializing entries.
        n_samples = len(global_bbox_id_list)
        n_nodes = 2 * n_samples - 1  # batch size
        A = np.empty(n_nodes, dtype=object)
        inertia = list()

        print(f'Constructing can not link for detections in the same image ... ')
        # start_time = time.time()
        mask = self.estimate_same_image_mask(global_bbox_id_list)
        # print("Constructing can not link for detections in the same image done in "
        #       "[--- %s seconds ---]" % (time.time() - start_time))

        start_time = time.time()
        for i in range(n_samples):
            row = np.where(mask[i, :] != 0)[0]  # np.where return a tuple
            data = pdist[i, row]
            A[i] = IntFloatDict(np.array(row, dtype=np.intp),
                                np.array(data, dtype=np.float64))  # fast dict
            # We keep only the upper triangular for the heap
            #  Generator expressions are faster than arrays on the following
            inertia.extend(_hierarchical.WeightedEdge(d, i, r)
                           for r, d in zip(row, data) if r > i)
        # This function accepts an arbitrary list and converts it to a heap
        # (sorted list)
        heapify(inertia)
        print("Initializing A and inertia done in  %s seconds ..."
              % (time.time() - start_time))
        print(f'len(inertia) = {len(inertia)} ... ')

        return inertia, A, pdist

    def _init_inertia_from_clusters(
            self, parent, children, used_node, pdist, clusters,
            linkage
    ):
        # Return a new array of given shape and type, without initializing entries.
        n_samples = pdist.shape[0]
        n_nodes = 2 * n_samples - 1  # batch size
        A = np.empty(n_nodes, dtype=object)
        inertia = list()

        start_time = time.time()
        labels = self.get_complete_node_list(parent=parent, n_samples=n_samples, used_node=used_node)
        num_clusters = len(labels)  # for this batch
        cluster_pdist_dict = {x: [] for x in labels}  # 2d array, first [] saves the rows, second the data.

        for k in range(num_clusters - 1):
            n_k = labels[k]
            rows = self.get_leaves(children=children, n_leaves=n_samples, node_id=n_k)

            for l in range(k + 1, num_clusters):
                n_l = labels[l]
                combined_cluster = clusters[n_k] + clusters[n_l]

                # -------------------------------- do we need to check image id? think in future
                image_ids = [x[0] for x in combined_cluster]
                # Detections xi and xj that are in the same images may be two isolated nodes
                # So they will be skipped if two clusters have overlap in image ids
                if not self.is_unique_list(image_ids):
                    continue
                # -------------------------------------------
                cols = self.get_leaves(children=children, n_leaves=n_samples, node_id=n_l)
                assert len(clusters[n_k]) > 0 and len(clusters[n_l]) > 0
                dist = AhcMetric.cluster_dist_from_pdist_rows_cols(
                    pdist=pdist, rows=rows, cols=cols,
                    linkage=linkage
                )
                # if appearance_dist >= 2:  # 1.0212946 ok, [-1,1] for similarity, 0,2 for distance 1 - simialrity
                #     print(f'self.temporal_distance(clusters[n_k], clusters[n_l]) ='
                #           f' {self.temporal_distance(clusters[n_k], clusters[n_l])}')
                assert dist <= 2.0

                cluster_pdist_dict[n_k].append([n_l, dist])
                cluster_pdist_dict[n_l].append([n_k, dist])
                # We keep only the upper triangular for the heap
                # Generator expressions are faster than arrays on the following
                if n_l < n_k:
                    inertia.append(_hierarchical.WeightedEdge(dist, n_l, n_k))
                else:
                    inertia.append(_hierarchical.WeightedEdge(dist, n_k, n_l))

        for i in labels:
            if len(cluster_pdist_dict[i]) == 0:
                A[i] = 0
            else:
                # 0 is the row, 1 is the dist
                A[i] = IntFloatDict(np.array([x[0] for x in cluster_pdist_dict[i]], dtype=np.intp),
                                    np.array([x[1] for x in cluster_pdist_dict[i]], dtype=np.float64))  # fast dict

        heapify(inertia)
        del cluster_pdist_dict, labels
        print("Initializing A and inertia done in %s ..." % (time.time() - start_time))
        return inertia, A

    def AgglomerativeTracking(
            self, global_bbox_id_list, distance_threshold, linkage,
            later_expert_flag=False
    ):
        """Conduct tracking for a given global_bbox_id_list. This function can be used as one-shoot tracking for a
        whole image sequence if it is short. The corresponding setting is global_bbox_id_list =

        :param global_bbox_id_list:
        :param distance_threshold:
        :param linkage:
        :param later_expert_flag:
        :return:
        """
        join_func = self._set_linkage_choice(linkage)

        # step 1: spatial-temporal constraint for detections in the same image
        n_samples = len(global_bbox_id_list)  # batch size
        n_nodes = 2 * n_samples - 1  # batch size

        if not later_expert_flag:
            self.local_parent_ = np.arange(n_nodes, dtype=np.intp)
            self.local_used_node_ = np.ones(n_nodes, dtype=np.intp)
            self.local_children_ = []
            self.local_distances_ = np.empty(n_nodes - n_samples)

            self.local_clusters_ = self.init_heap_clusters(global_bbox_id_list)
            inertia, A, self.local_pdist_ = self._init_inertia_from_scratch(
                global_bbox_id_list
            )
            # if pdist is None:
            #     # no need to set the same image id pair to BIG_VALUE as they will not be
            #     # considered (not pushed in inertia)
        else:
            inertia, A = self._init_inertia_from_clusters(
                parent=self.local_parent_, children=self.local_children_, used_node=self.local_used_node_,
                pdist=self.local_pdist_, clusters=self.local_clusters_, linkage=linkage
            )

        # step 2: recursive merge loop
        for k in range(2 * n_samples - len(self.local_clusters_), n_nodes):
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
                if self.local_used_node_[i] and self.local_used_node_[j]:
                    if self._is_valid_merging(self.local_clusters_[i], self.local_clusters_[j]):
                        break

            if len(inertia) == 0 or edge.weight > distance_threshold:
                break

            if (k - n_samples) % 100 == 0:
                print(f'{k - n_samples} th merging, len(inertia) = {len(inertia)}, '
                      f'len(clusters) = {len(self.local_clusters_)}, ({i}, {j}, {"%.2f" % edge.weight}) ...')

            # store the distance
            self.local_distances_[k - n_samples] = edge.weight

            self.local_parent_[i] = self.local_parent_[j] = k
            self.local_children_.append((i, j))
            # Keep track of the number of elements per cluster
            n_i = self.local_used_node_[i]  # number of observations
            n_j = self.local_used_node_[j]  # number of observations
            self.local_used_node_[k] = n_i + n_j
            self.local_used_node_[i] = self.local_used_node_[j] = False

            coord_col = join_func(A[i], A[j], self.local_used_node_, n_i,
                                  n_j)

            # transfer the detection id to global bbox id
            self.local_clusters_[k] = self.local_clusters_[i] + self.local_clusters_[j]
            heapify(self.local_clusters_[k])

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
            del self.local_clusters_[i], self.local_clusters_[j]
            # remove the key, value from the dict

        return self.local_clusters_


class AHCMultiSequenceTracker(AHC, AHCTracker):
    """Conduct all clustering based on pdist.
    Handling image sequence that is long so that we have to divide it into sub sequences.
    We assume all the detections can be load into memory for offline object tracking even we divide the sequence
    into sub sequences if the number of detections is too large so that generating pdist runs out of memory.
    So this class can not used for online tracking for videos with infinite length.
    """
    def __init__(self):
        super(AHCTracker, self).__init__()

    def single_expert_tracking(self, max_dist_thd, linkage, **kwargs):
        if 'num_frame_per_sub_seq' in kwargs:
            num_frame_per_sub_seq = kwargs['num_frame_per_sub_seq']
            D = self.auto_divide_sequence(
                num_frame_per_sub_seq=num_frame_per_sub_seq
            )
        else:
            if 'max_det_num_per_sub_seq' in kwargs:
                max_det_num_per_sub_seq = kwargs['max_det_num_per_sub_seq']
            else:
                max_det_num_per_sub_seq = self.batch_det_num
            D = self.auto_divide_sequence(
                max_det_num_per_sub_seq=max_det_num_per_sub_seq
            )

        for seq_id in range(len(D) - 1):  # len(D) - 1  range(2)
            global_bbox_id_list = D[seq_id] + D[seq_id + 1]
            # Skip sub sequence tracking if there is less than two detections
            # since no any merging is likely occur.
            if len(global_bbox_id_list) < 2:
                continue

            self.tracking_one_subsequent(
                distance_threshold=max_dist_thd,
                linkage=linkage,
                global_bbox_id_list=global_bbox_id_list, **kwargs
            )

