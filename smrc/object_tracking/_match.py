from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering

from object_tracking._metrics import *


# Hungarian method
# class Match:
def HungarianMatch(dist_matrix):
    """
    >>> import numpy as np
    >>> from scipy.optimize import linear_sum_assignment
    >>> cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    >>> cost
    array([[4, 1, 3],
           [2, 0, 5],
           [3, 2, 2]])
    >>> row_ind, col_ind = linear_sum_assignment(cost)
    >>> row_ind
    array([0, 1, 2])
    >>> col_ind
    array([1, 0, 2])
    >>> cost[row_ind, col_ind]
    array([1, 2, 2])
    :param dist_matrix:
    :return:
    """
    row_inds, col_inds = linear_sum_assignment(dist_matrix)
    return row_inds, col_inds


########################################
# BidirectionalBestMath
#########################################

# no need to keep this function, since we can get the BidirectionalMatch from dist_matrix
def BidirectionalMatchSimMat(
        similarity_matrix, min_similarity=1e-5
        ):
    """
    # similarity is used to generate recommendation level
    update the connectivity based on the estimated similarity.
    :param similarity_matrix: similarity value should between [0, 1]
    :param min_similarity:
    :return:
    """
    # check if the size of similarity_matrix > 0, and the value of similarity_matrix
    # in the correct range [0, 1]
    # For cosine similarity, the value could be negative.
    # So we need to properly handle it np.amin(similarity_matrix) >= 0
    assert similarity_matrix.size > 0 and np.amax(similarity_matrix) <= 1 \
        and np.amin(similarity_matrix) >= 0

    num_pre_det = similarity_matrix.shape[0]

    # sorts along second axis (cross) from small to large
    # each of the row is sorted
    row_sort_indices = np.argsort(similarity_matrix, axis=1)
    # extract the first column as the best match for the rows
    # matched_col_ind = row_sort_indices[-1, :]

    # sorts along first axis (down) from small to large
    # each of the column is sorted
    column_sort_indices = np.argsort(similarity_matrix, axis=0)
    # matched_row_ind = column_sort_indices[:, -1]

    row_inds = np.array(range(num_pre_det))
    col_inds = np.array([None] * num_pre_det)
    for i in range(len(num_pre_det)):
        j = row_sort_indices[i, -1]
        if column_sort_indices[-1, j] == i and \
                similarity_matrix[i, j] > min_similarity:
            col_inds[i] = j
    return _filter_out_none_matching(row_inds, col_inds)  # row_inds, col_inds


def BidirectionalMatch(
        dist_matrix,
        max_distance
):
    """
    >>> a = np.array([-1, None, 5])
    >>> np.where(a == None)
    (array([1]),)
    >>> np.where(a != None)
    (array([0, 2]),)
    :param dist_matrix:
    :param max_distance:
    :return:
    """
    assert dist_matrix.size > 0
    num_pre_det = dist_matrix.shape[0]

    # find the minimum value for each column, and each row
    # return a array, e.g., array([0, 0, 0])
    min_values_index_column = np.argmin(dist_matrix, axis=0)
    min_values_index_row = np.argmin(dist_matrix, axis=1)

    row_inds = np.array(range(num_pre_det))
    col_inds = np.array([None] * num_pre_det)
    for i in range(num_pre_det):
        j = min_values_index_row[i]
        # the min value of the distance should be smaller than inf_value
        # print(f'IoU = {dist_matrix[i, j]}')
        if min_values_index_column[j] == i and \
                dist_matrix[i, j] < max_distance:
            col_inds[i] = j
    return _filter_out_none_matching(row_inds, col_inds)


def _filter_out_none_matching(row_inds, col_inds):
    """
    Filter out None locations from col_inds.
        b = np.array([[1., 2., None], [np.nan, 4., 5.]])
        np.equal(b, None)
        Out[9]:
            array([[False, False, True],
                   [False, False, False]])
    :param row_inds: 1d array of type np.int32
    :param col_inds: 1d array of type np.int32
    :return:
    """
    # np.where(col_inds != None)  # deprecated
    valid_ids = np.not_equal(col_inds, None)
    # valid_ids = np.where(col_inds != None)  # deprecated but also works

    # We must transfer the dtype of col_inds to int, otherwise, its dtype is object
    # and can cause troubles for indexing, e.g., iou_mat[row_inds, col_inds]
    return row_inds[valid_ids], np.array(col_inds[valid_ids], dtype=np.int32)


def _frame_dist_mat_to_pairwise_dist_mat(dist_matrix):
    """
    https://stackoverflow.com/questions/44357591/assigning-values-to-a-block-in-a-numpy-array
    :param dist_matrix:
    :return:
    """
    n_row, n_col = dist_matrix.shape
    new_dim = n_row + n_col
    pairwise_dist_mat = np.zeros((new_dim, new_dim), dtype=float) + BIG_VALUE

    cols = range(n_row, new_dim)
    rows = range(n_row)
    pairwise_dist_mat[np.ix_(rows, cols)] = dist_matrix

    i_lower = np.tril_indices(new_dim, -1)
    pairwise_dist_mat[i_lower] = pairwise_dist_mat.T[i_lower]  # make the matrix symmetric
    return pairwise_dist_mat


def AHCMatch(dist_matrix, max_distance):
    # frame_dist_matrix to pairwise distance matrix
    pdist = _frame_dist_mat_to_pairwise_dist_mat(dist_matrix)

    # we must use the 'complete' or 'average' linkage, 'single' linkage is wrong.
    cluster = AgglomerativeClustering(
        n_clusters=None, linkage='complete',
        distance_threshold=max_distance,
        compute_full_tree=True,
        connectivity=None,
        affinity='precomputed'
    )
    cluster.fit(pdist)

    # transfer cluster._label to matching results
    num_pre_det = dist_matrix.shape[0]
    row_inds = np.array(range(num_pre_det))
    col_inds = np.array([None] * num_pre_det)
    labels = cluster.labels_
    for i in range(int(np.max(labels) + 1)):
        idx = np.where(labels == i)[0]
        # we must transfer the ndarry to list, otherwise, the cluster with 1 element will be saved as ndarry type
        # in the final clusters, while cluster with more than 1 element will be saved as list.
        # Different types in the list will cause problem in later processing
        if len(idx) < 2:  # np.min(idx) >= num_pre_det (not corrrect)
            continue
        else:
            col_inds[np.min(idx)] = np.max(idx) - num_pre_det

    return _filter_out_none_matching(row_inds, col_inds)


