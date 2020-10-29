import os
import json
import numpy as np
from smrc.utils.base import natural_sort_key, get_json_file_list
from .detection_process import nms_detection_dict, det_dict_to_smrc_tracking_format, \
    image_path_last_two_level

######################################################
# handle the results of yolo based object_detection
######################################################


def extract_smrc_json_dir(json_file_name):
    return json_file_name.replace('.json', '').replace('smrc_', '')


def get_json_dir_list(directory_path):
    json_files = get_json_file_list(directory_path, only_local_name=True)
    dir_list = [extract_smrc_json_dir(x) for x in json_files]
    return dir_list


def get_smrc_json_file_list(directory_path, only_local_name=False):
    # print(directory_path)
    file_path_list = []
    # load image list
    for f in sorted(os.listdir(directory_path), key=natural_sort_key):
        f_path = os.path.join(directory_path, f)
        # print(f_path)
        # print(f)
        if os.path.isdir(f_path):
            # skip directories
            continue
        elif os.path.isfile(f_path) and f.find('.json') > 0:
            # and f.startswith('smrc_')
            # f.find('smrc_') >= 0 also ok if not use ( f.startswith('smrc_') )
            if only_local_name:
                file_path_list.append(f)
            else:
                file_path_list.append(f_path)
    print(f'{len(file_path_list)} json files loaded with the format *.json.')
    return file_path_list


def load_json_detection_to_dict(
        json_detection_file, detection_dict=None,
        score_thd=None, nms_thd=None,
        short_image_path=True
):
    """Load the YOLO detections that are saved in json format to object_detection dict.

    :param json_detection_file: one file include the detections of one video, with the
        following format,
        [
            {"image_path":"/home/smrc/darknet/Detection_Taxi-raw-data_20200427_add2/62054/0000.jpg",
            "category_id":1, "bbox":[359, 220, 390, 236], "score":0.942105},
            ...
        ]
    :param detection_dict: if not none, continue to add the object_detection to the dict, this
        is useful for ensemble of multiple detections
    :param short_image_path: if true, only save the last two levels of the image path,
        i.e., 3440/0000.jpg
    :param score_thd: remove the low score object_detection, score < score_thd
    :param nms_thd: non maximum suppression threshold
    :return:
        object_detection dict with the format of
            [class_idx, xmin, ymin, xmax, ymax, score]
        A lot of public codes use the format of [class_idx, score, xmin, ymin, xmax, ymax],
        If we do need the public format, just conduct transformation.
    """
    # {'image_path': '/home/smrc/Data/1000videos/3440/0000.jpg', 'category_id': 2,
    # 'bbox': [275, 187, 78, 53], 'score': 0.984684}
    # print(f'Loading {json_detection_file} to detection_list...')
    with open(json_detection_file) as json_file:
        json_detection_data = json.load(json_file)

    # all the detections regarding this directory are saved here
    if detection_dict is None:
        detection_dict = {}

    count = 0
    for detection in json_detection_data:  # detection_idx,
        # load the image name
        image_path = detection['image_path']
        if short_image_path:
            image_path = image_path_last_two_level(image_path)

        class_idx = detection['category_id']
        xmin, ymin, xmax, ymax = list(map(int, detection['bbox']))
        score = detection['score']

        if score_thd is not None and score < score_thd:
            count += 1
            # print(f'    ignored object_detection {[class_idx, xmin, ymin, xmax, ymax, score]}...')
        else:
            if image_path in detection_dict:
                detection_dict[image_path].append(
                    [class_idx, xmin, ymin, xmax, ymax, score])
            else:
                detection_dict[image_path] = [
                    [class_idx, xmin, ymin, xmax, ymax, score]
                ]
    det_num = np.sum([len(dets) for key, dets in detection_dict.items()])
    print(f'    {count} object_detection ignored due to score_thd {score_thd}, '
          f'remaining {det_num} detections ...')

    if nms_thd is not None:
        detection_dict = nms_detection_dict(detection_dict, nms_thd)
    det_num_after_nms = np.sum([len(dets) for key, dets in detection_dict.items()])
    print(f'    {det_num - det_num_after_nms} object_detection ignored due to nms_thd {nms_thd}, '
          f'remaining {det_num_after_nms} detections ...')
    return detection_dict


def load_multiple_json_detection_to_dict(
        json_file_list, score_thd=None, nms_thd=None, short_image_path=True
):
    """# Ensemble multiple detection files
    """
    detection_dict = {}
    for json_file in json_file_list:
        detection_dict = load_json_detection_to_dict(
            json_detection_file=json_file,
            detection_dict=detection_dict,
            score_thd=score_thd,
            nms_thd=nms_thd,  # conduct nms in separate files as it should be
            short_image_path=short_image_path
        )

    # conduct nms again after detections from all files are loaded
    # Note that unexpected behavior may occur if the detections are from different
    # detectors and the scores are not normalized.
    if nms_thd is not None:
        detection_dict = nms_detection_dict(detection_dict, nms_thd)

    return detection_dict


#################################
# load data for tracking
#################################


def json_det_to_tracking_format(
        json_file, test_image_list, score_thd=None, nms_thd=None
):
    """
    [class_idx, x1, y1, x2, y2, score],
    :param json_file:
    :param test_image_list:
    :param nms_thd:
    :param score_thd:
    :return: a detection list of the format of [class_idx, x1, y1, x2, y2, score]
    """
    return load_json_det_files_to_tracking_format(
        json_file_list=[json_file],
        test_image_list=test_image_list,
        score_thd=score_thd,
        nms_thd=nms_thd
    )


def load_json_det_files_to_tracking_format(
        json_file_list, test_image_list,
        score_thd=None, nms_thd=None
):
    """
    load object_detection from json file to detection_list with non maximum suppression
    This function can be used for processing the object_detection of a single video, or
    the detections of multiple video as long as all the images are included in
    test_image_list
    :param json_file_list:
    :param test_image_list:
    :param nms_thd:
    :param score_thd:
    :return:
    """
    detection_dict = load_multiple_json_detection_to_dict(
        json_file_list, score_thd=score_thd, nms_thd=nms_thd,
        short_image_path=True
    )

    detection_list = det_dict_to_smrc_tracking_format(
        detection_dict, test_image_list
    )
    return detection_list
