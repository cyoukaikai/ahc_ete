import os
import sys
import smrc.utils
import numpy as np


def mAP(prediction, ground_truth, class_list, class_names=None, thds_of_interest=np.arange(0.05, 1, 0.05)):
    """
    estimate the mAP given predictions, and ground truth
    :param prediction: [
        [image_name, bbox_prediction]
    ], bbox_prediction is of format, [class_id, x1, y1, x2, y2, score]
    :param ground_truth:[
        [image_name, bbox_list]
    ]
    :param class_list: defined class ids, e.g., 0, 1, 2, ...
    :param class_names: corresponding class names for class list
    :return: the estimated mAP value and other important evaluation values
    """

    assert len(prediction) == len(ground_truth), \
        f'length of prediction {len(prediction)} and ground truth {len(ground_truth)} should be equal'

    # conduct non max suppression
    pred = non_max_suppression(prediction,thd=0.5)

    evaluation_result_list = []  # used to estimate the precision, recall, F1-score at different thd levels
    for idx in range(len(class_list)):
        class_id = class_list[idx]
        print(f'Extracting data for class {class_id}')

        # prepare data for class i
        pred_class_idx = extract_label_for_single_class(pred, class_id)
        ground_truth_class_idx = extract_label_for_single_class(pred, class_id)

        # estimate ap for class i
        # evaluation_results format [thd, TP, FP, FN, recall, IoU] (IoU is None, if not TP)
        # iou_list_for_single_class of different length with precisions, as it only conut
        # correct predictions
        evaluation_result = estimate_evaluation_metrics_for_single_class(pred_class_idx, ground_truth_class_idx)
        evaluation_result_list.append(evaluation_result)

    if thds_of_interest is not None:
        # report precision, recall, F1-score at different thd levels
        thds_of_interest = np.arange(0.05, 1, 0.05)
        estimate_ap_wrt_thd_of_interest(
            evaluation_result_list, thds_of_interest
        )

    # ap for each class
    ap_list = estimate_ap_wrt_recall(evaluation_result_list, class_list, class_names)
    mAP_result = np.mean(ap_list)

    return mAP_result


def non_max_suppression(raw_prediction, thd=0.5):
    pred = []
    for image_pred in raw_prediction:
        raw_bbox_array = np.array(image_pred)
        boxes, scores = raw_bbox_array[:, 1:5], raw_bbox_array[:, 5:].flatten()

        if len(image_pred) > 1:
            selected = smrc.utils.non_max_suppression(boxes, scores, thd)
            bbox_processed = raw_bbox_array[selected,0:5].astype("int").tolist()
        else:
            bbox_processed = raw_bbox_array[:,0:5].astype("int").tolist()

        pred.append(bbox_processed)
    return pred


def estimate_evaluation_metrics_for_single_class(pred_class_idx,
                                                 ground_truth_class_idx
                                                 ):
    """

    :param pred_class_idx:
    :param ground_truth_class_idx:
    :return:
    evaluation_results format [thd, TP, FP, FN, recall, IoU] (IoU is None, if not TP)
    """

    evaluation_result = []
    assert len(pred_class_idx) == len(ground_truth_class_idx)

    pred_flattened = []
    for image_id, image_pred in enumerate(pred_class_idx):
        if len(image_pred) > 0:
            for item in image_pred:
                # item is a list, [class_id, x1, y1, x2, y2, score]
                pred_flattened.append([image_id] + item)

    # sort the pred_flattened
    sorted(pred_flattened, key=lambda x: x[6], reverse=True)

    detection_available = [ [] for x in pred_class_idx]
    evaluation_dict = {}
    TP, FP, FN, IoU = 0, 0, 0, 0
    for rank_id, single_pred in enumerate(pred_flattened):
        print(f'rank {rank_id}')

        image_id = single_pred[0]
        # bbox_rect = single_pred[2:6]
        score = single_pred[6]

        detection_available[image_id].append(single_pred)
        if image_id in evaluation_dict:
            tp_old, fp_old, fn_old, IoU_old = evaluation_dict[image_id]
        else:
            tp_old, fp_old, fn_old = 0, 0, 0

        tp, fp, fn, ious = estimate_evaluation_metrics_single_image(
            detection_available[image_id], ground_truth_class_idx[image_id]
        )
        evaluation_dict[image_id] = [tp, fp, fn, ious]
        tp_diff, fp_diff, fn_diff = tp - tp_old, fp - fp_old, fn - fn_old
        TP += tp_diff
        FP += fp_diff
        FN += fn_diff
        IoU += np.sum(ious - IoU_old)
        recall = estimate_recall(TP=TP, FN=FN)
        evaluation_result.append([score, TP, FP, FN, recall, IoU])

    return evaluation_result
    # record = []
    # for idx, pred in enumerate(pred_class_idx):
    #     ground_truth = ground_truth_class_idx[idx]
    #     tps, fps, fns, IoUs = estimate_evaluation_metrix_single_image(pred, ground_truth)
    #     for i in range(len(tps)):
    #         score = pred[i][5]
    #         record.append([tps[i], fps[i], fns[i], score, IoUs[i]])

    # # sort the record based on its confidence level
    #
    #
    # for item in record:
    #     tp, fp, fn, thd, IoU = item
    #     TP += tp
    #     FP += fp
    #     FN += fn
    #     recall = estimate_recall(TP=TP, FN=FN)
    #     evaluation_result.append([thd, TP, FP, FN, recall, IoU])
    #
    # return evaluation_result


def estimate_evaluation_metrics_single_image(preds, ground_truth):

    tp, fp, fn, IoUs = 0, 0, 0, 0
    if len(preds) > 0:
        pred_used = [False] * len(preds)
        iou_thd = 0.5
        for ground_truth_bbox_rect in ground_truth:
            max_iou = 0
            correct_pred_idx = None
            for i, pred in enumerate(preds):
                bbox_rect = pred[1:5]
                iou = smrc.utils.compute_iou(bbox_rect, ground_truth_bbox_rect)
                if iou > max_iou:
                    max_iou = iou
                    correct_pred_idx = i

            if max_iou >= iou_thd:
                tp += 1
                pred_used[correct_pred_idx] = True
                IoUs.append(max_iou)
            else:
                fn += 1

        for i in range(len(preds)):
            if pred_used[i] is False:
                fp += 1

    return tp, fp, fn, IoUs


def extract_label_for_single_class(pred, class_id):
    resulting_pred_list = []
    for frame_preds in pred:
        # the first element for a prediction is the class id
        frame_pred_filtered = [x for x in frame_preds if x[0] == class_id]
        resulting_pred_list.append(frame_pred_filtered)

    return resulting_pred_list


def estimate_ap_wrt_recall(evaluation_result_list, class_list, class_names):
    # start 0.1, stop 1, step 0.1
    #array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ])
    recall_of_interest = np.arange(0.1, 1.1, 0.1)

    ap_list = []
    for recall in recall_of_interest:
        TP_list, FP_list, FN_list = [], [], []
        precision_list, recall_list, IoU_list, F1_score_list = [], [], [], []

        for idx in range(len(class_list)):
            class_id = class_list[idx]
            if class_names is not None:
                print(f'Estimating precision, recall, F1-score for class {class_id} ({class_names[idx]}) ')
            else:
                print(f'Estimating precision, recall, F1-score for class {class_id}')

            # find the index of the row of precisions to extract
            evaluation_result = evaluation_result_list[idx]
            recall_index = len(evaluation_result) - 1

            # [thd, TP, FP, FN, precision, recall, IoU](IoU is None, if not TP)
            for i, item in enumerate(evaluation_result):
                v = item[5]
                if v < recall:
                    recall_index = i - 1
                    break

            # print(f'thd = {thd}, thd_index = {thd_index}')
            thd_tmp, TP, FP, FN = evaluation_result[recall_index][:4]

            precision, recall_estimated = estimate_precision_recall(TP, FP, FN)
            F1_score = estimate_F1_score(precision, recall_estimated)

            ious = [x[5] for x in evaluation_result[:recall_index + 1] if x[5] is not None]
            avg_iou = np.mean(ious)
            print(f'class {class_id}, recall {recall}, thd {thd_tmp}, precision {precision}, '
                  f'recall_estimated {recall_estimated}, '
                  f'F1_score {F1_score}, avg_iou = {avg_iou}')

            precision_list.append(precision)
            recall_list.append(recall)
            IoU_list.append(avg_iou)
            F1_score_list.append(F1_score)

            TP_list.append(TP)
            FP_list.append(FP)
            FN_list.append(FN)
            print(f'class {class_id}, TP_list = {TP_list}')
            print(f'class {class_id}, FP_list = {FP_list}')
            print(f'class {class_id}, FN_list = {FN_list}')
            print(f'class {class_id}, FN_list = {IoU_list}')

        TP_sum, FP_sum, FN_sum = np.sum(TP_list), np.sum(FP_list), np.sum(FN_list)
        precision, recall = estimate_precision_recall(TP_sum, FP_sum, FN_sum)
        F1_score = estimate_F1_score(precision, recall)
        print('========================================= (not important)')
        print(f'averaged over all classes with equal weight for each prediction')
        print(f'for recall {recall}, class {class_id},  TP_sum {TP_sum}, FP_sum {FP_sum}, '
              f'FN_sum {FN_sum}, F1_score = {F1_score}')

        average_iou = np.mean(IoU_list)
        ap = np.mean(precision_list)
        ap_list.append(ap)

        print('========================================= (important)')
        print(f'averaged over all classes with equal weight for each class')
        print(f'for recall {recall},  precision {precision}, recall {recall}, F1_score {F1_score}.')
        print(f'average IoU {average_iou}')

    return ap_list


def estimate_ap_wrt_thd_of_interest(evaluation_result_list, thds_of_interest,
                                    class_list, class_names):
    for thd in thds_of_interest:
        TP_list, FP_list, FN_list= [], [], []
        ap_list, recall_list, IoU_list, F1_score_list  = [], [], [], []

        for idx in range(len(class_list)):
            class_id = class_list[idx]
            if class_names is not None:
                print(f'Estimating precision, recall, F1-score for class {class_id} ({class_names[idx]}) ')
            else:
                print(f'Estimating precision, recall, F1-score for class {class_id}')

            # find the index of the row of precisions to extract
            evaluation_result = evaluation_result_list[idx]
            thd_index = len(evaluation_result) - 1
            for i, item in enumerate(evaluation_result):
                v = item[0]
                if v < thd:
                    thd_index = i - 1
                    break

            # print(f'thd = {thd}, thd_index = {thd_index}')
            thd_tmp, TP, FP, FN, _ = evaluation_result[thd_index][:4]

            precision, recall =estimate_precision_recall(TP, FP, FN)
            F1_score = estimate_F1_score(precision, recall)

            ious = [x[6] for x in evaluation_result[:thd_index+1] if x[6] is not None]
            avg_iou = np.mean(ious)
            print(f'class {class_id}, thd {thd}, precision {precision}, recall {recall}, '
                  f'F1_score {F1_score}, avg_iou = {avg_iou}')

            ap_list.append(precision)
            recall_list.append(recall)
            IoU_list.append(avg_iou)
            F1_score_list.append(F1_score)

            TP_list.append(TP)
            FP_list.append(FP)
            FN_list.append(FN)
            print(f'class {class_id}, TP_list = {TP_list}')
            print(f'class {class_id}, FP_list = {FP_list}')
            print(f'class {class_id}, FN_list = {FN_list}')
            print(f'class {class_id}, FN_list = {IoU_list}')

        TP_sum, FP_sum, FN_sum = np.sum(TP_list), np.sum(FP_list), np.sum(FN_list)
        precision, recall = estimate_precision_recall(TP_sum, FP_sum, FN_sum)
        F1_score = estimate_F1_score(precision, recall)
        print('========================================= (not important)')
        print(f'averaged over all classes with equal weight for each prediction')
        print(f'for thd {thd} class {class_id}, thd {thd}, TP_sum {TP_sum}, FP_sum {FP_sum}, '
              f'FN_sum {FN_sum}, F1_score = {F1_score}')

        average_iou = np.mean(IoU_list)
        print('========================================= (important)')
        print(f'averaged over all classes with equal weight for each class')
        print(f'for thd {thd} precision {precision}, recall {recall}, F1_score {F1_score}.')
        print(f'average IoU {average_iou}')


def estimate_F1_score(precision, recall):
    return 2 * precision * recall / (precision + recall)


def estimate_precision_recall(TP, FP, FN):
    precision = estimate_precision(TP, FP)
    recall = estimate_recall(TP, FN)
    return precision, recall


def estimate_precision(TP, FP):
    # https://en.wikipedia.org/wiki/Precision_and_recall
    return TP / (TP + FP)


def estimate_recall(TP, FN):
    # https://en.wikipedia.org/wiki/Precision_and_recall
    return TP / (TP + FN)


def load_class_list(class_name_file):
    if not os.path.isfile(class_name_file):
        print('File {} not exist, please check.'.format(class_name_file))
        sys.exit(0)

    # load class list
    print(f'Loading class list from file {class_name_file}')
    with open(class_name_file) as f:
       class_name_list = list(smrc.utils.non_blank_lines(f))
    f.close()  # close the file

    class_list = list[range(len(class_name_list))]

    return class_list, class_name_list


def load_detection_from_file(ann_path):
    detection_list = []
    if os.path.isfile(ann_path):
        # edit YOLO file
        with open(ann_path, 'r') as old_file:
            lines = old_file.readlines()
        old_file.close()

        # print('lines = ',lines)
        for line in lines:
            result = line.split(' ')

            # the data format in line (or txt file) should be int type, 0-index.
            # we transfer them to int again even they are already in int format (just in case they are not)
            bbox = [int(result[0]), int(result[1]), int(result[2]), int(
                result[3]), int(result[4]), float(result[5])]
            detection_list.append(bbox)
    return detection_list


def load_prediction_from_txt_root_dir(test_image_path, image_dir, label_dir):
    assert len(test_image_path) > 0

    # each image corresponds to one list in prediction
    prediction_all = []
    for image_path in test_image_path:
        ann_path = smrc.utils.get_image_or_annotation_path(image_path, image_dir,
                                                           label_dir, '.txt')
        if not os.path.isfile(ann_path):
            detection = []
        else:
            detection = load_detection_from_file(ann_path)

        prediction_all.append(detection)

    return prediction_all


def load_ground_truth_from_txt_root_dir(test_image_path, image_dir, ground_truth_dir):
    assert len(test_image_path) > 0

    # each image corresponds to one list in prediction
    prediction_all = []
    for image_path in test_image_path:
        ann_path = smrc.utils.get_image_or_annotation_path(image_path, image_dir,
                                                           ground_truth_dir, '.txt')
        if not os.path.isfile(ann_path):
            detection = []
        else:
            detection = smrc.utils.load_bbox_from_file(ann_path)

        prediction_all.append(detection)

    return prediction_all


def load_test_image_list(filename):
    return smrc.utils.load_1d_list_from_file(filename)


if __name__ == "__main__":

    IMAGE_DIR = 'images'
    LABEL_DIR = 'labels'
    GROUND_TRUTH_DIR = ''
    CLASS_LIST_FILE = 'class_list.txt'
    TEST_IMAGE_LIST_FILE = ''

    test_image_list = load_test_image_list(TEST_IMAGE_LIST_FILE)
    prediction = load_prediction_from_txt_root_dir(test_image_list, IMAGE_DIR, LABEL_DIR)
    ground_truth = load_ground_truth_from_txt_root_dir(test_image_list, IMAGE_DIR, GROUND_TRUTH_DIR)

    thds_of_interest = None #np.arange(0.05, 1, 0.05)
    class_list, class_names = load_class_list(CLASS_LIST_FILE)

    mAP(prediction, ground_truth, class_list, class_names, thds_of_interest)
