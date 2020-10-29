from smrc.object_tracking import Tracking
from smrc.io_smrc.detection_process  import *


class DetectionToAnnotation(Tracking):
    def __init__(self):
        self.name = 'DetectionToAnnotation'
        # self.detection_score_available = False

    def preprocess_json_detection_for_annotation(
            self, video_json_file, video_image_list, video_label_dir_to_save,
            detection_accept_thd, detection_to_be_decided_thd
    ):

        # load json object_detection with score greater than detection_to_be_decided_thd
        detection_dict = load_json_detection_to_dict(
            video_json_file, short_image_path=True
        )
        # a list of detection_list, i.e., [detection_list0, detection_list1, ..., ]
        # detection_list has the format of [ [class_idx, xmin, ymin, xmax, ymax, score],  ... ]
        # one image, one object_detection list (empty in the case of no any object_detection results)
        video_detection_list = det_dict_to_tracking_det_list (
            detection_dict, video_image_list, nms_thd=0.5
        )

        # initialize self.video_detected_bbox_all, self.frame_dets, self.connectedness
        # and so on, if no object_detection exists, then object_tracking tool will quit here
        self.init_tracking_tool(video_detection_list, detection_to_be_decided_thd)

        # conduct offline object_tracking for the object_detection

        # loop through each cluster, if no any bbox in the cluster has score greater
        # than detection_accept_thd, than delete this cluster.

        # save all the bbox remained in the clusters.




    def offline_tracking(self, video_detection_list, cluster_dist_thd=3, delete_outlier=False):

        if len(self.video_detected_bbox_all) == 0:
            return self.clusters, self.video_detected_bbox_all

        # connect the bbox in the neighboring images
        self.ExpertTeam_ConnectNeighbouringFrames()

        # self._extract_cluster_based_on_connectivity()

        self.ExpertTeam_ConnectShortBreak(max_break_length=cluster_dist_thd)

        self._extract_cluster_based_on_connectivity()

        # self.Expert_EuclideanDist_ConnectNeighbouringFrames()

        # ExpertTeam_ConnectShortBreak(max_break_length=5)
        #
        # ExpertTeam_ConnectLongBreak()
        #
        # ExpertTeam_FillMissedDetection()
        #
        # ExpertTeam_IdentifyConsistentFalsePositive()
        #
        # ExpertTeam_DeleteFalsePositiveShortBreak()

        # self.connect_clusters_end_points(
        #     cluster_dist_thd=cluster_dist_thd,
        #     curve_fitting=False
        # )
        # # print('Beginning to connect trajectory ... ')
        # # self.connect_clusters_end_points(
        # #     cluster_dist_thd=cluster_dist_thd,
        # #     curve_fitting=True
        # # )
        #
        # # for cluster_id, cluster in enumerate(self.clusters):
        # #     self.record_single_cluster_for_debugging(self.clusters, cluster_id, 'round1')
        #
        # # self.connect_cluster_Agglomerative()
        #
        # # sys.exit(0)
        # # self.estimate_display_object_id()
        # if delete_outlier:
        #     self.delete_cluster_with_length_thd()

        # self.cluster_sorted_index = sorted(range(len(self.clusters)),
        #                                    key=lambda k: len(self.clusters[k]),
        #                                    reverse=True)  # sort the list based on its first element
        #
        # self.cluster_labels = self.estimate_cluster_label(self.clusters)
        #
        # self.estimate_display_object_id(self.CLASS_LIST)

        return self.clusters, self.video_detected_bbox_all
#
# #[.50:.05:.95]
# def mAP_VOC(
#         prediction, ground_truth, interpolation_method='ElevenPointInterpolation',
#         class_names=None
# ):
#     """
#     estimate the mAP given predictions, and ground truth
#     This script requires the format of object_detection, [class_id, x1, y1, x2, y2, score]
#         1. load_detection_normal_format_from_file(ann_path):
#             bbox = [int(result[0]), int(result[2]), int(
#                 result[3]), int(result[4]), int(result[5]), float(result[1])]
#         2. evaluate_single_image(preds_sorted, ground_truth, iou_thd=0.5):
#             iou = smrc.line.compute_iou(pred[1:5], ground_truth_bbox[1:5])
#         3. def evaluate_single_class(pred_class_idx, ground_truth_class_idx):
#             scores = [x[5] for x in image_pred]
#         are all designed to fit this format
#
#     :param prediction: object_detection after non maximum suppression
#     format [
#         detection_list
#     ], each bbox in bbox_list is of format, [class_id, x1, y1, x2, y2, score]
#     e.g., detection_list =
#     <class 'list'>: [
#         [2, 105, 166, 122, 206, 0.024659], [1, 350, 187, 393, 217, 0.982402], ...
#     ]
#     :param ground_truth:
#         a list of bbox_list, [ bbox_list0, ...    ]
#         e.g.,
#         <class 'list'>: [[2, 79, 240, 130, 344]]
#     :param interpolation_method: 'ElevenPointInterpolation', 'EveryPointInterpolation'
#     :param class_list: defined class ids, e.g., 0, 1, 2, ...
#     :param class_names: corresponding class names for class list
#     :return: the estimated mAP value and ap for each class
#     """
#
#     evaluation_result_list, count_ground_truth_bbox_dict = evaluate_all_class(
#         prediction, ground_truth
#     )
#     # ap for each class
#     # ElevenPointInterpolation EveryPointInterpolation
#     ap_list = estimate_ap(evaluation_result_list, interpolation_method)
#     print('==========================================')
#     print(f'ap_list = {ap_list}')
#     mAP_result = np.mean(ap_list)
#     print(f'mAP_result = {mAP_result}')
#
#     return mAP_result
#
#
# def estimate_ap(evaluation_result_list, interpolation_method):
#     """
#     :param evaluation_result_list: a list of evaluation_result for each class
#     :param class_list:
#     :param class_names:
#     :param interpolation_method:
#     :return:
#     """
#     AP_list = []
#     for class_id in range(len(evaluation_result_list)):
#         print(f'Estimating precision, recall, F1-score for class {class_id}')
#
#         # find the index of the row of precisions to extract
#         evaluation_result = evaluation_result_list[class_id]
#         precision_values = [x[3] for x in evaluation_result]
#         recall_values = [x[4] for x in evaluation_result]
#
#         if interpolation_method == 'ElevenPointInterpolation':
#             precision_list, recall_list, F1_score_list, Iou_list = eleven_point_interpolation(
#                 precision_values, recall_values
#             )
#             AP = np.mean(precision_list)
#         elif interpolation_method == 'EveryPointInterpolation':
#             precision_list, recall_list = every_point_interpolation(
#                 precision_values, recall_values
#             )
#
#             AP = precision_list[0] * recall_list[0]
#             for idx in range(1, len(precision_list)):
#                 AP += precision_list[idx] * (recall_list[idx] - recall_list[idx - 1])
#
#         AP_list.append(AP)
#     return AP_list
#
#
# def average_precision_recall_wrt_thd(
#         prediction, ground_truth, thds_of_interest=None
# ):
#     """
#     report precision, recall, F1-score at different thd levels
#     :param prediction:
#     :param ground_truth:
#     :param thds_of_interest:
#     :return:
#     """
#
#     evaluation_result_list, count_ground_truth_bbox_dict = evaluate_all_class(
#         prediction, ground_truth
#     )
#
#     # initialize thds_of_interest if it is not specified.
#     if thds_of_interest is None:
#         # thds_of_interest = [0.5, 0.25, 0.24, 0.2, 0.15]
#         # thds_of_interest = list(np.arange(1, -0.05, -0.05))
#         thds_of_interest = list(np.linspace(0, 1, 21))
#     print(f'thds_of_interest = {thds_of_interest}')
#
#
#     for thd in thds_of_interest:
#         TP_list, FP_list, FN_list, IoU_list = [], [], [], []
#         ap_list, recall_list, avg_IoU_list, F1_score_list  = [], [], [], []
#
#         for class_id in range(len(evaluation_result_list)):
#             # class_id = class_list[idx]
#
#             # find the index of the row of precisions to extract
#             evaluation_result = evaluation_result_list[class_id]
#
#             score_values = [x[0] for x in evaluation_result]
#             precision, recall, F1_score, avg_iou = None, None, None, None
#             TP_accumulated, FP_accumulated, FN_accumulated = 0, 0, count_ground_truth_bbox_dict[class_id]
#             Iou_accumulated = 0
#
#             # print(f'thd = {thd}')
#             # print(f'score_values = {score_values}')
#             index_all = np.argwhere(np.array(score_values[:]) >= thd)
#             # index_all = np.argwhere(score_values[:] >= thd)
#             if index_all.size == 0:
#                 print(f'thd {thd}, Upper bound: {evaluation_result[0]}')
#             else:
#                 score_index = index_all.max()
#                 thd_tmp, TP_accumulated, FP_accumulated, precision, recall, Iou_accumulated = \
#                     evaluation_result[score_index]
#
#                 F1_score = estimate_F1_score(precision, recall)
#                 avg_iou = Iou_accumulated / TP_accumulated
#
#                 FN_accumulated = count_ground_truth_bbox_dict[class_id] - TP_accumulated
#
#                 eval_annotation = 0.9 * recall + 0.1 * precision
#                 print(f'[class]{class_id}, [thd]{"{0:.2f}".format(thd)},  '
#                       f'[TP]{str(TP_accumulated).ljust(6)}, '
#                       f'[FP]{"{:05d}".format(FP_accumulated)}, '
#                       f'[FN]{"{:05d}".format(FN_accumulated)},'
#                       f'[prec]{"{0:.4f}".format(precision)}, '
#                       f'[rec]{"{0:.4f}".format(recall)}, '
#                       f'[eval]{"{0:.4f}".format(eval_annotation)}')
#             # 'hi'.ljust(10)
#             # "{:05d}".format(FP_accumulated)
#             ap_list.append(precision)
#             recall_list.append(recall)
#             avg_IoU_list.append(avg_iou)
#             F1_score_list.append(F1_score)
#
#             # TP_list.append(TP_accumulated)
#             # FP_list.append(FP_accumulated)
#             # FP_list.append(FN_accumulated)
#             # IoU_list.append(Iou_accumulated)
#
#         # no need to do anything if we have only one class
#         if len(evaluation_result_list) > 1:
#             print('========================================= (important)')
#             print(f' thd {thd}: averaged over all classes with equal weight for each class')
#
#             mean_ap_over_class = sum(filter(None, ap_list))
#             print(f'ap_list = {ap_list}, {len(ap_list) - ap_list.count(None)} valid elements.')
#             print(f'mean_ap_over_class = {mean_ap_over_class}')
#
#             mean_recall_over_class = sum(filter(None, recall_list))
#             print(f'recall_list = {recall_list}, {len(recall_list) - recall_list.count(None)} valid elements.')
#             print(f'mean_recall_over_class = {mean_recall_over_class}')
#
#             # mean_F1_score_over_class = sum(filter(None, F1_score_list))
#             # print(f'F1_score_list = {F1_score_list}, {len(F1_score_list) - F1_score_list.count(None)} valid elements.')
#             # print(f'mean_F1_score_over_class = {mean_F1_score_over_class}')
#             #
#             # mean_avg_IoU_over_class = sum(filter(None, avg_IoU_list))
#             # print(f'IoU_list = {avg_IoU_list}, {len(avg_IoU_list) - avg_IoU_list.count(None)} valid elements.')
#             # print(f'mean_avg_IoU_over_class = {mean_avg_IoU_over_class}')
#
#             # print('========================================= (not important)')
#             # TP_sum, FP_sum, FN_sum = np.sum(TP_list), np.sum(FP_list), np.sum(FN_list)
#             # precision, recall = estimate_precision_recall(TP_sum, FP_sum, FN_sum)
#             # F1_score = estimate_F1_score(precision, recall)
#             # iou_sum = sum(IoU_list)
#             # print(f'averaged over all classes with equal weight for each prediction')
#             # print(f'for thd {thd} class {class_id}, thd {thd}, '
#             #       f'TP_sum {TP_sum}, FP_sum {FP_sum}, '
#             #       f'FN_sum {FN_sum}, '
#             #       f'precision {"{0:.4f}".format(precision)}, '
#             #       f'recall {"{0:.4f}".format(recall)}, '
#             #       f'F1_score {"{0:.4f}".format(F1_score)}, '
#             #       f'Iou_sum = {"{0:.2f}".format(iou_sum)}')
#
#
#
# # def evaluate_prediction_without_score(prediction, ground_truth, class_list, class_names=None, thds_of_interest=None):
# #     """
# #     estimate the mAP given predictions, and ground truth
# #     :param prediction: [
# #         [image_name, bbox_prediction]
# #     ], bbox_prediction is of format, [class_id, x1, y1, x2, y2, score]
# #     :param ground_truth:[
# #         [image_name, bbox_list]
# #     ]
# #     :param class_list: defined class ids, e.g., 0, 1, 2, ...
# #     :param class_names: corresponding class names for class list
# #     :return: the estimated mAP value and other important evaluation values
# #     """
# #
# #     assert len(prediction) == len(ground_truth), \
# #         f'length of prediction {len(prediction)} and ground truth {len(ground_truth)} should be equal'
# #
# #     evaluation_result_list = []  # used to estimate the precision, recall, F1-score at different thd levels
# #     count_ground_truth_bbox_dict = {}
# #     for idx in range(len(class_list)):
# #         class_id = class_list[idx]
# #
# #         # prepare data for class i
# #         print(f'Extracting object_detection and ground truth data for class {class_id}')
# #         pred_class_idx = extract_detection_or_truth_for_single_class(prediction, class_id)
# #         ground_truth_class_idx = extract_detection_or_truth_for_single_class(ground_truth, class_id)
# #
# #
# #         # estimate ap for class i
# #         # evaluation_results format [thd, TP_accumulated, FP_accumulated, precision, recall, IoU_accumulated]
# #         (IoU is None, if not TP)
# #         # iou_list_for_single_class of different length with precisions, as it only conut
# #         # correct predictions
# #         print(f'Estimating TP, FP for class {class_id}')
# #         print(f'{len(pred_class_idx)} object_detection files, {len(ground_truth_class_idx)} ground truth files')
# #         evaluation_result, num_ground_truth_bbox = evaluate_single_class(pred_class_idx, ground_truth_class_idx)
# #         evaluation_result_list.append(evaluation_result)
# #         count_ground_truth_bbox_dict[class_id] = num_ground_truth_bbox
# #     # # ap for each class
# #     # ap_list = estimate_ap(evaluation_result_list, class_list, class_names,
# #     #     interpolation_method = 'ElevenPointInterpolation'
# #     # )
# #     #
# #     # #ElevenPointInterpolation EveryPointInterpolation
# #     # mAP_result = np.mean(ap_list)
# #     # print(f'mAP_result = {mAP_result}')
# #
# #
# #     if thds_of_interest is None:
# #         # report precision, recall, F1-score at different thd levels
# #         # thds_of_interest = [0.25]
# #         thds_of_interest = list(np.arange(1, -0.05, -0.05))
# #
# #     estimate_ap_wrt_thd_of_interest(
# #         evaluation_result_list, count_ground_truth_bbox_dict,
# #         thds_of_interest, class_list, class_names
# #     )
# #
# #     # return mAP_result
#
#     # conduct non max suppression
#     # pred = non_max_suppression(prediction, thd=0.5)
#     # pred = prediction
#     # generate_txt_file_for_public_map_code(pred,ground_truth)
#
#
# def evaluate_all_class(prediction, ground_truth):
#     """
#     estimate tp, fp, fn given predictions, and ground truth
#     :param prediction: object_detection after non maximum suppression
#     format [
#         detection_list
#     ], each bbox in bbox_list is of format, [class_id, x1, y1, x2, y2, score]
#     e.g., detection_list =
#     <class 'list'>: [
#     [2, 105, 166, 122, 206, 0.024659], [1, 350, 187, 393, 217, 0.982402], ...
#     ]
#     :param ground_truth:
#         a list of bbox_list, [ bbox_list0, ...    ]
#         e.g.,
#         <class 'list'>: [[2, 79, 240, 130, 344]]
#     :param class_list: defined class ids, e.g., 0, 1, 2, ...
#     :return: a list of evaluation_results with the format
#         [thd, TP_accumulated, FP_accumulated, precision, recall, IoU_accumulated]
#         (IoU is None, if not TP)
#         Each evaluation_result corresponding to one class
#     """
#
#     assert len(prediction) == len(ground_truth), \
#         f'length of prediction {len(prediction)} and ground truth {len(ground_truth)} should be equal'
#
#     pred = sort_prediction_for_each_image_based_on_score(prediction)
#     class_list_tmp = []
#     for bbox_list in ground_truth:
#         if bbox_list is not None:
#             class_list_tmp.extend([bbox[0] for bbox in bbox_list])
#
#     class_list = list(set(class_list_tmp))
#     print(f'unique class list = {class_list}')
#
#     evaluation_result_list = []  # used to estimate the precision, recall, F1-score at different thd levels
#     count_ground_truth_bbox_dict = {}
#     for idx in range(len(class_list)):
#         class_id = class_list[idx]
#
#         # prepare data for class i
#         print('==================================================')
#         print(f'Extracting object_detection and ground truth data for class {class_id}')
#         pred_class_idx = extract_data_single_class(pred, class_id)
#         ground_truth_class_idx = extract_data_single_class(ground_truth, class_id)
#
#         # estimate ap for class i
#         print(f'Estimating TP, FP for class {class_id}')
#         print(f'{len(pred_class_idx)} object_detection files, {len(ground_truth_class_idx)} ground truth files')
#
#         # evaluation_results format
#         #   [thd, TP_accumulated, FP_accumulated, precision, recall, IoU_accumulated] (IoU is None, if not TP)
#         evaluation_result, num_ground_truth_bbox = evaluate_single_class(
#             pred_class_idx, ground_truth_class_idx
#         )
#         evaluation_result_list.append(evaluation_result)
#         count_ground_truth_bbox_dict[class_id] = num_ground_truth_bbox
#
#     return evaluation_result_list, count_ground_truth_bbox_dict
#
#
# def evaluate_single_class(pred_class_idx, ground_truth_class_idx):
#     """
#     evaluate the metrics for a single class
#     :param pred_class_idx: object_detection for single class, a list of bbox_list
#     :param ground_truth_class_idx: ground truth for single class, a list of bbox list
#         the order of the ground truth are consistent with the order of the object_detection,
#         and correspond to the same image.
#     :return:
#         evaluation_results format
#         [thd, TP, FP, precision, recall, IoU] (IoU is None, if object_detection is not TP)
#     """
#     evaluation_result = []
#     assert len(pred_class_idx) == len(ground_truth_class_idx)
#
#     print('Estimating TP, FP, IOU ...')
#     pred_flattened = []
#     for image_id, image_pred in enumerate(pred_class_idx):
#         if len(image_pred) > 0:
#             scores = [x[5] for x in image_pred]
#             tp_list, fp_list, iou_list = evaluate_single_image(
#                 image_pred, ground_truth_class_idx[image_id]
#             )
#             pred_flattened.extend(list(zip(scores, tp_list, fp_list, iou_list)))
#
#
#     # # sys.exit(0)
#     # # sort the pred_flattened
#     print(f'Sorting results based on scores, {len(pred_flattened)} predicted bboxes...')
#     pred_flattened = sorted(pred_flattened, key=lambda x: x[0], reverse=True)
#
#     num_ground_truth_bbox = 0
#     for x in ground_truth_class_idx:
#         num_ground_truth_bbox += len(x)
#
#     TP_accumulated, FP_accumulated, IoU_accumulated = 0, 0, 0
#     TP_all = [x[1] for x in pred_flattened]
#     FP_all = [x[2] for x in pred_flattened]
#     IoU_all = [x[3] for x in pred_flattened]
#
#     print(f'tp = {np.sum(TP_all)}, fp = {np.sum(FP_all)}, fn = {num_ground_truth_bbox - np.sum(TP_all)}')
#     # smrc.line.save_1d_list_to_file('tp_list.txt',TP_all )
#     # print(f'fp in total = {np.sum(FP_all)}')
#     # smrc.line.save_1d_list_to_file('fp_list.txt',FP_all )
#     # TP_accumulated_tmp = np.cumsum(TP_all)
#     # smrc.line.save_1d_list_to_file('TP_accumulated_tmp.txt',TP_accumulated_tmp )
#     # print(f'fn in total = {num_ground_truth_bbox - np.sum(TP_all)}')
#
#     # sys.exit(0)
#     print('Estimating precision_values, recall_values for the sorted results ...')
#     for rank_id, single_pred in enumerate(pred_flattened):
#         # print(f'rank {rank_id}')
#         score, tp, fp, iou = single_pred
#         TP_accumulated = np.sum(TP_all[:rank_id+1])
#         FP_accumulated = np.sum(FP_all[:rank_id+1])
#         FN_accumulated = num_ground_truth_bbox - TP_accumulated
#         IoU_accumulated = np.sum([x for x in IoU_all[:rank_id+1] if x is not None])
#         precision,recall = estimate_precision_recall(TP=TP_accumulated,
#                                                      FP=FP_accumulated,
#                                                      FN=FN_accumulated
#                                                      )
#
#         evaluation_result.append([score, TP_accumulated, FP_accumulated,
#                                   precision, recall, IoU_accumulated])
#
#     return evaluation_result, num_ground_truth_bbox
#
#
# def evaluate_single_image(preds_sorted, ground_truth, iou_thd=0.5):
#     preds = preds_sorted
#     tp_list, fp_list, IoU_list = [0] * len(preds), [0] * len(preds), [None] * len(preds)
#
#     if len(preds) > 0:
#         ground_truth_used = [False] * len(ground_truth)
#         for pred_id, pred in enumerate(preds):
#             max_iou = 0
#             fitted_ground_truth_idx = None
#             for i, ground_truth_bbox in enumerate(ground_truth):
#                 # only select the prediction from unused ones
#                 if ground_truth_used[i]: # True
#                     continue
#                 iou = smrc.line.compute_iou(pred[1:5], ground_truth_bbox[1:5])
#                 # print(f'prediction bbox = {pred[1:5]}, score = {pred}')
#                 # print(f'ground truth bbox = {ground_truth_bbox[1:5]}')
#                 # print(f'iou = {iou}')
#                 if iou > max_iou:
#                     max_iou = iou
#                     fitted_ground_truth_idx = i
#             # print(f'max_iou = {max_iou}')
#             if max_iou >= iou_thd:
#                 tp_list[pred_id] = 1
#                 ground_truth_used[fitted_ground_truth_idx] = True
#                 IoU_list[pred_id] = max_iou
#                     # fp_list[correct_pred_idx] = 1
#                     # pred_used[correct_pred_idx] = True
#             else:
#                 fp_list[pred_id] = 1
#     # print(f'tp_list = {tp_list}')
#     return tp_list, fp_list, IoU_list
#
#
# def sort_prediction_for_each_image_based_on_score(prediction_all):
#     print('Sorting predictions for each image ...')
#     prediction_all_sorted = []
#     for item in prediction_all:
#         if len(item) == 0:
#             prediction_all_sorted.append([])
#         else:
#             item_sorted = sorted(item, key=lambda x: x[-1], reverse=True)
#             prediction_all_sorted.append(item_sorted)
#
#     return prediction_all_sorted
#
#
# def extract_data_single_class(detection_or_truth, class_id):
#     """
#     extract the object_detection or ground truth data for a single class
#     The resulting data format of object_detection or ground truth are the same with
#     the input format.
#     :param detection_or_truth: an ordered list of bbox_list
#     :param class_id: the class id of interest
#     :return:
#         extracted data
#     """
#     resulting_detection_or_truth_list = []
#     for frame_detection_or_truth in detection_or_truth:
#         # the first element for a prediction is the class id
#         frame_detection_or_truth_filtered = [x for x in frame_detection_or_truth if x[0] == class_id]
#         resulting_detection_or_truth_list.append(frame_detection_or_truth_filtered)
#
#     return resulting_detection_or_truth_list
#
#
# def eleven_point_interpolation(precision_values, recall_values):
#     # TP_list, FP_list = [], []
#     assert len(precision_values) == len(recall_values)
#
#     recall_of_interest = list(np.linspace(0, 1, 11))
#
#     precision_list, recall_list, F1_score_list, IoU_list = [], [], [], []
#     for recall in recall_of_interest:
#         # recall_index = len(evaluation_result) -1 # default value
#         # [thd, TP, FP, FN, precision, recall, IoU](IoU is None, if not TP)
#         recall_index = None
#         # The precision at each recall level r is interpolated by taking
#         # the maximum precision measured for a method for which
#         # the corresponding recall exceeds r:
#
#         precision = 0
#         index_all = np.argwhere(recall_values[:] >= recall)
#         if index_all.size != 0:
#             # recall_index = index_all.min()
#             precision = max(precision_values[index_all.min():])
#
#         # for i, v in enumerate(recall_values):
#         #     if recall <= v:
#         #         recall_index = i
#         #         break
#         #
#         # if recall_index is None:
#         #     precision = 0
#         # else:
#         #     precision = np.max(precision_values[recall_index:])
#
#         F1_score = estimate_F1_score(precision, recall)
#         print(f'recall {recall},  precision {precision}, '
#                 f'F1_score {F1_score}')
#         precision_list.append(precision)
#         recall_list.append(recall)
#         F1_score_list.append(F1_score)
#
#     return precision_list, recall_list, F1_score_list, IoU_list
#
#
# def every_point_interpolation(precision_values, recall_values):
#     assert len(precision_values) == len(recall_values)
#     precision_list, recall_list = [], []
#
#     init_index = 0
#     # print(f'precision_values={precision_values},  recall_values {recall_values}')
#     while True:
#         max_prec = max(precision_values[init_index:])
#         index_all = np.argwhere(precision_values[:] == max_prec)
#         index = index_all.max()
#
#         # print(f'max_prec = {max_prec}, index_all = {index_all}, index = {index}')
#         # sys.exit(0)
#         precision, recall = precision_values[index], recall_values[index]
#         # F1_score = estimate_F1_score(precision, recall)
#         precision_list.append(precision)
#         recall_list.append(recall)
#
#
#         init_index = index + 1
#         if index == len(precision_values) - 1:
#             break
#     # sys.exit(0)
#
#     return precision_list, recall_list
#
#
# def estimate_F1_score(precision, recall):
#     return 2 * precision * recall / (precision + recall)
#
#
# def estimate_precision_recall(TP, FP, FN):
#     precision = estimate_precision(TP, FP)
#     recall = estimate_recall(TP, FN)
#     return precision,recall
#
#
# def estimate_precision(TP, FP):
#     # https://en.wikipedia.org/wiki/Precision_and_recall
#     return TP / (TP + FP)
#
#
# def estimate_recall(TP, FN):
#     # https://en.wikipedia.org/wiki/Precision_and_recall
#     return TP / (TP + FN)
#
#
# # ==================================================
# # not used
# def get_iou(a, b, epsilon=1e-5):
#     """ Given two boxes `a` and `b` defined as a list of four numbers:
#             [x1,y1,x2,y2]
#         where:
#             x1,y1 represent the upper left corner
#             x2,y2 represent the lower right corner
#         It returns the Intersect of Union score for these two boxes.
#
#     Args:
#         a:          (list of 4 numbers) [x1,y1,x2,y2]
#         b:          (list of 4 numbers) [x1,y1,x2,y2]
#         epsilon:    (float) Small value to prevent division by zero
#
#     Returns:
#         (float) The Intersect of Union score.
#     """
#     # COORDINATES OF THE INTERSECTION BOX
#     x1 = max(a[0], b[0])
#     y1 = max(a[1], b[1])
#     x2 = min(a[2], b[2])
#     y2 = min(a[3], b[3])
#
#     # AREA OF OVERLAP - Area where the boxes intersect
#     width = (x2 - x1)
#     height = (y2 - y1)
#     # handle case where there is NO overlap
#     if (width<0) or (height <0):
#         return 0.0
#     area_overlap = width * height
#
#     # COMBINED AREA
#     area_a = (a[2] - a[0]) * (a[3] - a[1])
#     area_b = (b[2] - b[0]) * (b[3] - b[1])
#     area_combined = area_a + area_b - area_overlap
#
#     # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
#     iou = area_overlap / (area_combined+epsilon)
#     return iou
#
#
# # not used
# def batch_iou_vec(a, b, epsilon=1e-5):
#     """ Given two arrays `a` and `b` where each row contains a bounding
#         box defined as a list of four numbers:
#             [x1,y1,x2,y2]
#         where:
#             x1,y1 represent the upper left corner
#             x2,y2 represent the lower right corner
#         It returns the Intersect of Union scores for each corresponding
#         pair of boxes.
#
#     Args:
#         a:          (numpy array) each row containing [x1,y1,x2,y2] coordinates
#         b:          (numpy array) each row containing [x1,y1,x2,y2] coordinates
#         epsilon:    (float) Small value to prevent division by zero
#
#     Returns:
#         (numpy array) The Intersect of Union scores for each pair of bounding
#         boxes.
#     """
#     # COORDINATES OF THE INTERSECTION BOXES
#     x1 = np.array([a[:, 0], b[:, 0]]).max(axis=0)
#     y1 = np.array([a[:, 1], b[:, 1]]).max(axis=0)
#     x2 = np.array([a[:, 2], b[:, 2]]).min(axis=0)
#     y2 = np.array([a[:, 3], b[:, 3]]).min(axis=0)
#
#     # AREAS OF OVERLAP - Area where the boxes intersect
#     width = (x2 - x1)
#     height = (y2 - y1)
#
#     # handle case where there is NO overlap
#     width[width < 0] = 0
#     height[height < 0] = 0
#
#     area_overlap = width * height
#
#     # COMBINED AREAS
#     area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
#     area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
#     area_combined = area_a + area_b - area_overlap
#
#     # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
#     iou = area_overlap / (area_combined + epsilon)
#     return iou
#
#
# def load_prediction_from_txt_files(label_dir):
#     ann_path_list = smrc.line.get_file_list_recursively(label_dir)
#     prediction_all = []
#     for ann_path in ann_path_list:
#         object_detection = load_detection_normal_format_from_file(ann_path)
#         # each file corresponding to one element in prediction_all
#         prediction_all.append(object_detection)
#     return prediction_all
#
#
# def load_detection_normal_format_from_file(ann_path):
#     detection_list = []
#     if os.path.isfile(ann_path):
#         # edit YOLO file
#         with open(ann_path, 'r') as old_file:
#             lines = old_file.readlines()
#         old_file.close()
#
#         # print('lines = ',lines)
#         for line in lines:
#             result = line.split(' ')
#
#             # the data format in line (or txt file) should be int type, 0-index.
#             # we transfer them to int again even they are already in int format (just in case they are not)
#             bbox = [int(result[0]), int(result[2]), int(
#                 result[3]), int(result[4]), int(result[5]), float(result[1])]
#             detection_list.append(bbox)
#
#     return detection_list
#
#
# def load_ground_truth_from_txt_files(label_dir):
#     ann_path_list = smrc.line.get_file_list_recursively(label_dir)
#     prediction_all = []
#     for ann_path in ann_path_list:
#         object_detection = smrc.line.load_bbox_from_file(ann_path)
#         # each file corresponding to one element in prediction_all
#         prediction_all.append(object_detection)
#
#     return prediction_all
#
#
# def load_class_list(class_name_file):
#     if not os.path.isfile(class_name_file):
#         print('File {} not exist, please check.'.format(class_name_file))
#         sys.exit(0)
#
#     # load class list
#     print(f'Loading class list from file {class_name_file}')
#     with open(class_name_file) as f:
#        class_name_list = list(smrc.line.non_blank_lines(f))
#     f.close()  # close the file
#
#     class_list = list(range(len(class_name_list)))
#
#     return class_list, class_name_list
#
#
# def tests():
#     LABEL_DIR = 'datasets/detections'
#     prediction = load_prediction_from_txt_files(LABEL_DIR)
#
#     GROUND_TRUTH_DIR = 'datasets/groundtruths'
#     ground_truth = load_ground_truth_from_txt_files(GROUND_TRUTH_DIR)
#
#     # CLASS_LIST_FILE = 'class_list.txt'
#     # class_list, class_names = load_class_list(CLASS_LIST_FILE)
#
#     # ElevenPointInterpolation
#     mAP_VOC(prediction, ground_truth,
#             interpolation_method='EveryPointInterpolation')
#     average_precision_recall_wrt_thd(prediction, ground_truth)
#
#
#
# def load_ground_truth_files_based_on_image_list(ground_truth_dir, test_image_path):
#     assert len(test_image_path) > 0
#
#     # each image corresponds to one list in prediction
#     prediction_all = []
#     for idx, image_path in enumerate(test_image_path):
#         image_dir, dir_name, image_name = smrc.line.split_image_path(image_path)
#
#         # file_names = image_path.split(os.path.sep)
#         # pos = image_path.find(file_names[-2])
#         # image_dir = image_path[:pos-1]
#         # print(f'image_dir = {image_dir}')
#         ann_path = smrc.line.get_image_or_annotation_path(
#             image_path, image_dir, ground_truth_dir, '.txt'
#         )
#
#         if not os.path.isfile(ann_path):
#             object_detection = []
#         else:
#             object_detection = smrc.line.load_bbox_from_file(ann_path)
#
#         prediction_all.append(object_detection)
#
#     return prediction_all
#
