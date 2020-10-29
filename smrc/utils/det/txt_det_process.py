# handle json detections of smrc format
import os
import cv2
import numpy as np
import json
import sys

import smrc.utils
from .detection_process import *


def load_raw_detection_to_bbox_list(ann_path, score_thd=None, score_position='last'):
    """
    load detections that have score > score_thd from app_path
    object_detection could have two formats specified by score_position
    :param ann_path:
    :param score_thd: default None, i.e., 0, loading all detections
    :param score_position:
        'last', [class_idx, x1, y1, x2, y2, score]
        'second', [class_idx, score, x1, y1, x2, y2]
    :return:
    """
    assert_score_position(score_position)
    if score_thd is None: score_thd = 0

    bbox_list = []
    if os.path.isfile(ann_path):
        with open(ann_path, 'r') as old_file:
            lines = old_file.readlines()
        old_file.close()

        # print('lines = ',lines)
        for line in lines:
            result = line.split('\n')[0].split(' ')
            class_idx, x1, y1, x2, y2, score = parse_det(result, score_position)
            assert 0 <= score <= 1, 'object_detection score is not in the range of [0,1]'
            if score > score_thd:
                bbox_list.append([class_idx, x1, y1, x2, y2])
        # print(f'bbox_list={bbox_list}')
    return bbox_list


def extract_raw_detection_to_bbox_list(image_dir, label_dir, score_position, score_thd=None, nms_thd=None):
    """
    Filtering out low score object_detection (score <= score_thd), discard score from
    [class_idx, x1, y1, x2, y2, score] and save
    to bbox list [class_idx, x1, y1, x2, y2]
    # ===================================================
    # typical use: face extraction from the object_detection results of tiny face network
    # =================================================
    :param image_dir: use to check if the object_detection is valid
        examples of invalid object_detection: 0 0.9103323480077511 -2.0 281.0 85.0 408.0
    :param label_dir: where the detections are saved
    :param score_thd:
    :param score_position
    :return:
    """
    dir_list = smrc.utils.get_dir_list_in_directory(label_dir)
    # the output of the result
    result_dir = label_dir + str(score_thd)

    for idx, dir_name in enumerate(dir_list):
        print(f'Processing {os.path.join(label_dir, dir_name)}, [{idx+1}/{len(dir_list)}] ...')
        ann_path_list = smrc.utils.get_file_list_recursively(
            os.path.join(label_dir, dir_name)
        )
        smrc.utils.generate_dir_if_not_exist(
            os.path.join(result_dir, dir_name)
        )

        # get the image information for a video
        image_path_list = smrc.utils.get_file_list_recursively(
            os.path.join(image_dir, dir_name)
        )
        assert len(image_path_list) > 0
        image_path = image_path_list[0]
        # check if it is an image
        test_img = cv2.imread(image_path)
        assert test_img is not None
        img_height, img_width = test_img.shape[:2]

        # load the bbox from object_detection files
        for ann_path in ann_path_list:
            bbox_list = load_raw_detection_to_bbox_list(ann_path, score_thd=score_thd, score_position=score_position)

            # conduct non max suppression if nms_thd is specified
            if nms_thd is not None:
                assert 0 <= nms_thd <= 1, 'nms_thd is not in the range of [0,1]'
                bbox_list = non_max_suppression_single_image(
                    image_pred=bbox_list, score_position=score_position, nms_thd=nms_thd
                )
            # the object_detection may include invalid results (e.g., x1, x2 < 0, or > image_width)
            # so we need to check
            bbox_list = smrc.utils.post_process_bbox_list(
                bbox_list, image_height=img_height,
                image_width=img_width
            )
            if len(bbox_list) > 0:
                result_ann_path = ann_path.replace(label_dir, result_dir)
                smrc.utils.save_bbox_to_file(result_ann_path, bbox_list)


def test_face_extraction():
    image_dir = 'Truck-sampleData114videos'  # 'image-inside-car'
    label_dir = 'resutls'  # resutls
    score_thd = 0.5
    extract_raw_detection_to_bbox_list(
        image_dir, label_dir,
        score_thd=score_thd, score_position='second'
    )


def load_any_format_detection_from_file(ann_path):
    """
    Only load the data, not doing any parsing for the data format
    :param ann_path:
    :return:
    """
    detection_list = []
    if os.path.isfile(ann_path):
        # edit YOLO file
        with open(ann_path, 'r') as old_file:
            lines = old_file.readlines()
        old_file.close()

        # print('lines = ',lines)
        for line in lines:
            result = line.split(' ')
            det = [float(result[0]), float(result[1]), float(result[2]), float(
                result[3]), float(result[4]), float(result[5])]
            detection_list.append(det)
    return detection_list


def load_txt_detection_to_dict(
        image_sequence_dir,
        detection_dir_name,
        score_position,
        video_image_list=None,
        detection_dict=None,
        score_thd=None,
        short_image_path=True
):
    """
    video_image_list is primary if you want specify the images to object_tracking, for instance, online object_tracking
    Otherwise, both video_image_list and image_sequence_dir are OK (equally priority)
        if video_image_list is None, then load images from image_sequence_dir
        if video_image_list is specified and included all images, then save as loading images from image_sequence_dir

    :param image_sequence_dir:
    :param detection_dir_name:
    :param score_position:
    :param video_image_list:
    :param detection_dict:
    :param score_thd:
    :param short_image_path:
    :return:
    """
    video_image_list = load_test_image_list_if_not_specified(
        image_sequence_dir=image_sequence_dir, video_image_list=video_image_list
    )
    if score_thd is None: score_thd = 0

    # {'image_path': '/home/smrc/Data/1000videos/3440/0000.jpg', 'category_id': 2,
    # 'bbox': [275, 187, 78, 53], 'score': 0.984684}
    print(f'Loading detections in {detection_dir_name} for {len(video_image_list)} images to detection_list...')

    # all the detections regarding this directory are saved here
    if detection_dict is None:
        detection_dict = {}

    for image_path in video_image_list:
        ann_path = smrc.utils.get_image_or_annotation_path(image_path, image_sequence_dir,
                                                           detection_dir_name, '.txt')
        if os.path.isfile(ann_path):
            detection_list = load_any_format_detection_from_file(
                ann_path
            )
            if len(detection_list) > 0:
                if short_image_path:
                    image_path = smrc.utils.image_path_last_two_level(image_path)

                for det in detection_list:
                    class_idx, xmin, ymin, xmax, ymax, score = parse_det(det=det, score_position=score_position)
                    if score >= score_thd:
                        if image_path in detection_dict:
                            detection_dict[image_path].append(
                                [class_idx, xmin, ymin, xmax, ymax, score])
                        else:
                            detection_dict[image_path] = [
                                [class_idx, xmin, ymin, xmax, ymax, score]
                            ]
                    else:
                        print(f'load_txt_detection_to_dict: ignored object_detection {[class_idx, xmin, ymin, xmax, ymax, score]}...')

    return detection_dict


def load_multiple_txt_detection_dir_to_dict(
    txt_det_file_dir_list, image_sequence_dir, video_image_list=None,
    score_thd=0.05, score_position=ScorePosition.Second,
    short_image_path=True
):
    video_image_list = load_test_image_list_if_not_specified(
        image_sequence_dir=image_sequence_dir, video_image_list=video_image_list
    )
    detection_dict = {}
    for txt_dir in txt_det_file_dir_list:
        detection_dict = load_txt_detection_to_dict(
            image_sequence_dir=image_sequence_dir,
            video_image_list=video_image_list,
            detection_dir_name=txt_dir,
            score_position=score_position,
            detection_dict=detection_dict,
            score_thd=score_thd,
            short_image_path=short_image_path
        )
    return detection_dict


def load_txt_detection_files_with_score_and_nms_thd(
        txt_det_file_dir_list, image_sequence_dir, video_image_list=None,
        score_thd=None, score_position=ScorePosition.Second,
        nms_thd=None, with_image_path=False
):
    video_image_list = load_test_image_list_if_not_specified(
        image_sequence_dir=image_sequence_dir, video_image_list=video_image_list
    )

    detection_dict = load_multiple_txt_detection_dir_to_dict(
        image_sequence_dir=image_sequence_dir,
        video_image_list=video_image_list,
        txt_det_file_dir_list=txt_det_file_dir_list,
        score_position=score_position,
        score_thd=score_thd,
        short_image_path=True
    )

    detection = det_dict_to_tracking_det_list(
        detection_dict, image_list=video_image_list,
        nms_thd=nms_thd, with_image_path=with_image_path
    )

    return detection


def load_test_image_list_if_not_specified(image_sequence_dir, video_image_list=None):
    if video_image_list is None:
        video_image_list = smrc.utils.get_file_list_recursively(image_sequence_dir)
    return video_image_list
