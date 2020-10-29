import os
# import cv2
import smrc.utils
import numpy as np
import sys
from tqdm import tqdm
#############################################################
#  general functions used for processing the detections
############################################################


class ScorePosition:
    Second = 'second'  # [class_idx, score, x1, y1, x2, y2]
    Last = 'last'  # [class_idx, x1, y1, x2, y2, score]

    DetScoreIndex = {
            Second: 1,  # [class_idx, x1, y1, x2, y2, score]
            Last: -1  # [class_idx, score, x1, y1, x2, y2]
    }


def get_det_index(score_position):
    return ScorePosition.DetScoreIndex[score_position]


def get_score(det, score_position):
    return det[get_det_index(score_position)]


def parse_det(det, score_position):
    assert_score_position(score_position)
    class_idx, x1, y1, x2, y2, score = None, None, None, None, None, None
    det = list(map(float, det))
    if score_position == ScorePosition.Last:
        class_idx, x1, y1, x2, y2, score = int(det[0]), round(det[1]), round(det[2]), \
                                           round(det[3]), round(det[4]), float(det[5])
    elif score_position == ScorePosition.Second:
        class_idx, score, x1, y1, x2, y2 = int(det[0]), float(det[1]), round(det[2]), \
                                           round(det[3]), round(det[4]), round(det[5])

    return class_idx, x1, y1, x2, y2, score


def assert_score_position(score_position):
    assert score_position in [ScorePosition.Last, ScorePosition.Second]


def non_max_suppression(boxes, scores, overlap_thresh):
    """
    http://pynote.hatenablog.com/entry/opencv-non-maximum-suppression
    input:boxes, array of size n by 4, n means the number of bboxes,
                    each row has the format of [x1, y1, x2, y2]
        scores, array of size n by 1,
        overlap_thresh, the bboxes overlap with the higher score bbox will be suppressed.

    usage example,
        raw_bbox_array = np.array(raw_bbox) #list to array, raw_bbox = [class_idx, x1, y1, x2, y2, score]
        boxes, scores = raw_bbox_array[:, 1:5], raw_bbox_array[:, 5:].flatten()

        if len(raw_bbox) > 1: #if more than one bbox
            selected = non_max_suppression(boxes, scores, overlap_thd)
            bbox_processed = raw_bbox_array[selected,0:5].astype("int").tolist()
        else:
            bbox_processed = raw_bbox_array[:,0:5].astype("int").tolist()
    """
    if boxes.size == 0:
        return []

    boxes = boxes.astype("float")
    x1, y1, x2, y2 = np.squeeze(np.split(boxes, 4, axis=1))

    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    indices = np.argsort(scores)
    selected = []

    while len(indices) > 0:
        last = len(indices) - 1

        selected_index = indices[last]
        remaining_indices = indices[:last]
        selected.append(selected_index)

        i_x1 = np.maximum(x1[selected_index], x1[remaining_indices])
        i_y1 = np.maximum(y1[selected_index], y1[remaining_indices])
        i_x2 = np.minimum(x2[selected_index], x2[remaining_indices])
        i_y2 = np.minimum(y2[selected_index], y2[remaining_indices])

        i_w = np.maximum(0, i_x2 - i_x1 + 1)
        i_h = np.maximum(0, i_y2 - i_y1 + 1)

        overlap = (i_w * i_h) / area[remaining_indices]

        indices = np.delete(indices, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

    # return boxes[selected].astype("int")

    # return the indices of the boxes rather than the boxes as we need to save the
    # corresponding class ids
    return selected


def non_max_suppression_single_image(image_pred, nms_thd, score_position='last'):
    """
    :param image_pred: a detection_list
    :param nms_thd: non max suppression threshold
    :param score_position:
    :return:
    """
    selected = non_max_suppression_selected_index(
        image_pred=image_pred, nms_thd=nms_thd, score_position=score_position
    )
    return [image_pred[x] for x in selected]


def non_max_suppression_selected_index(image_pred, nms_thd, score_position='last'):
    """
    :param image_pred: a detection_list
    :param nms_thd: non max suppression threshold
    :param score_position:
    :return:
    """
    if len(image_pred) <= 1:
        return list(range(len(image_pred)))
    else:
        raw_bbox_array = np.array(image_pred)

        assert_score_position(score_position)

        boxes, scores = None, None
        if score_position == ScorePosition.Last:
            # for object_detection format: [class_id, x1, y1, x2, y2, score]
            boxes, scores = raw_bbox_array[:, 1:5], raw_bbox_array[:, -1].flatten()
        elif score_position == ScorePosition.Second:
            # for object_detection format: [class_id, score, x1, y1, x2, y2]
            boxes, scores = raw_bbox_array[:, 2:], raw_bbox_array[:, 1].flatten()

        selected = non_max_suppression(boxes, scores, nms_thd)
        # bbox_processed = raw_bbox_array[selected, :].tolist()

        return selected


######################################################################
# handle the results of yolo based object_detection of the smrc json format
######################################################################

def det_dict_to_det_list(
        detection_dict, with_image_path=False
    ):
    detection_list_all_frames = []

    for image_path in sorted(detection_dict):
        if with_image_path:
            detection_list_all_frames.append(
                [image_path, detection_dict[image_path]]
            )
        else:
            detection_list_all_frames.append(detection_dict[image_path])

    return detection_list_all_frames


def nms_detection_dict(detection_dict, nms_thd):
    det_num = count_det_num(detection_dict)
    for image_path in detection_dict:
        detection_dict[image_path] = non_max_suppression_single_image(
            detection_dict[image_path], nms_thd
        )
    det_num_after_nms = count_det_num(detection_dict)
    print(f'{det_num - det_num_after_nms} object_detection ignored due to nms_thd {nms_thd}, '
          f'remaining {det_num_after_nms} detections ...')
    return detection_dict

######################################################
# transfer detection data to object tracking format
##################################################


def image_path_last_two_level(image_path):
    file_names = image_path.split(os.path.sep)
    assert len(file_names) >= 2
    return os.path.join(file_names[-2], file_names[-1])


def det_dict_to_smrc_tracking_format(
        detection_dict, image_list,
        score_thd=None,
        nms_thd=None
):
    return det_dict_to_tracking_det_list(
        detection_dict, image_list,
        score_thd, nms_thd,
        with_image_path=True
    )


def det_dict_to_tracking_det_list(
    detection_dict, image_list,
    score_thd=None,
    nms_thd=None,
    with_image_path=True
):
    """Only used for smrc object tracking interface
    load object_detection dict to a list of object_detection list, one image one object_detection list.
    There may be images with no object_detection, or detections with multiple videos in a single
    dict, by specifying the image_list, we can assign empty list to images and extract the
    detections of a single video for object object_tracking.
    :param detection_dict: two level image path, e.g., '239412/0003.jpg'
    :param image_list: image path list of at least two levels,
        e.g.,
        '239412/0003.jpg' or
        '/home/smrc/darknet/data/smrc/images/1000videos/153311/0162.jpg'
        if the images in image_list are only partial of the key (image_path) of detection_dict,
        then only the corresponding image_list are extracted.
    :param nms_thd: conduct nms with given threshold
    :param with_image_path: include or not include image path in the final
        object_detection list. If true, the image path will be saved in
        the resulting object_detection list
    :param score_thd: filter out low score detections
    :return: detection_list_all_frames
        format
        if with_image_path is False
            [detection_list0, detection_list1, ..., ]
            , where detection_list has the format of
            [
               [class_idx, x1, y1, x2, y2, score],
                ...
            ]
        if with_image_path is True
            [
                [image_path, detection_list0],
                [image_path, detection_list1],
                ...,
            ], where detection_list has the format of
                [
                   [class_idx, x1, y1, x2, y2, score],
                    ...
                ]
    """
    assert image_list is not None and len(image_list) > 0

    detection_list_all_frames = []
    for idx, image_path_to_check in enumerate(image_list):
        # extract the last two levels of the image path
        image_path = image_path_last_two_level(image_path_to_check)
        detection = []
        if image_path in detection_dict:
            if score_thd is not None:
                detection_dict = [det for det in detection_dict[image_path]
                                  if det[-1] >= score_thd]

            if nms_thd is not None:
                detection = non_max_suppression_single_image(
                    detection_dict[image_path], nms_thd
                )
            else:
                detection = detection_dict[image_path]

        if with_image_path:
            detection_list_all_frames.append([image_path_to_check, detection])
        else:
            detection_list_all_frames.append(detection)

    return detection_list_all_frames


######################################################
# operation for object_detection list
#######################################################
def count_det_num(detection_dict):
    return np.sum([len(dets) for key, dets in detection_dict.items()])


def filter_out_low_score_detection_dict(detection_dict, score_thd):
    new_dict = {}
    for key, det_bbox_list in detection_dict.items():
        det_bbox_list_filtered = filter_out_low_score_det(det_bbox_list, score_thd)
        new_dict[key] = det_bbox_list_filtered
    count1, count2 = count_det_num(detection_dict), count_det_num(new_dict)
    print(f'{count1 - count2} object_detection ignored due to score_thd {score_thd}, '
          f'remaining {count2} detections ...')
    return new_dict


def filter_out_low_score_det(detection_list, score_thd, det_idx=-1):
    # we assume the score is the last column
    return [det for det in detection_list if det[det_idx] <= score_thd]


def filter_out_detection_score_thd(
        sequence_detections, low_score_thd, high_score_thd, score_position='last'):
    """
    Remaining detections should have score >= low_score_thd, <= high_score_thd.
    :param sequence_detections: a list of detection_list, each of which corresponding to the object_detection
    of a image
        e.g., [class_idx, x1, y1, x2, y2, score]
        or [class_idx, score, x1, y1, x2, y2]

    :param low_score_thd:
    :param high_score_thd:
    :param score_position: use to specify the index of the confidence score in the object_detection
    :return:
    """

    assert_score_position(score_position)
    det_idx = get_det_index(score_position)

    remaining_detection = []
    for detection_list in sequence_detections:
        filtered_detection_list = [det for det in detection_list
                                   if low_score_thd <= det[det_idx] <= high_score_thd]
        remaining_detection.append(filtered_detection_list)

    return remaining_detection


def save_detection_list_to_file(ann_path, detection_list, score_position_to_save='last'):
    """Save the object_detection list to file with the format specified by score_position_to_save
    detection_list, the detections for a single image of the format of
    [
        [class_idx, xmin, ymin, xmax, ymax, score],
        ...
    ]
    """
    # save the bbox if it is not the active bbox (future version, active_bbox_idxs includes more than one idx)
    assert_score_position(score_position_to_save)

    with open(ann_path, 'w') as new_file:
        for detection in detection_list:
            # the detction has a fixed format through the smrc application
            class_idx, xmin, ymin, xmax, ymax, score = detection
            items = None
            if score_position_to_save == ScorePosition.Last:
                items = map(str, [int(class_idx), round(xmin), round(ymin),
                                  round(xmax), round(ymax), score])
            elif score_position_to_save == ScorePosition.Second:
                items = map(str, [int(class_idx), score, round(xmin), round(ymin),
                                  round(xmax), round(ymax)])
            assert items is not None
            txt_line = ' '.join(items)
            # we need to add '\n'(newline), otherwise, all the bboxes will be in one line and not able to be recognized.
            new_file.write(txt_line + '\n')
    new_file.close()

# def nms_detection_list()


# def det_list_to_bbox_list(image_det):
#     return [det[:5] for det in image_det]

