# data transfermation for mAP estimation
import os
import sys
import numpy as np
import json
import shutil
import cv2

import smrc.utils


# # ways to load the prediction and ground truth
# # a) load object_detection based on ground truth files
# #   (one ground truth file one object_detection file, if no object_detection file, then it is
# #   regarded as empty object_detection)

# # b) load object_detection and ground truth based on image lists.
# def load_det_json_file_truth_txt_based_on_txt_image_list(
#         test_image_list, detection_json_file,
# ):

# def load_det_json_dir_truth_txt_based_on_image_dir(

# ):


# def load_det_json_based_on_truth_txt():


# def load_det_txt_based_on_truth_txt():


# def load_det_txt_truth_txt(option=)
#     'all_txt'
#     'non_empty_truth_file'
#     'non_empty_detection_file'




    # video_detection_list = []
    # for dir_name in dir_list:
    #     json_file = os.path.join(json_dir, 'smrc_' + dir_name + '.json')
    #     print(f'{json_file} ...')
    #
    #
    #     image_dir = os.path.join(image_root_dir, dir_name)
    #     test_image_list = smrc.line.get_file_list_recursively(
    #         image_dir  # , ext_str='.jpg'
    #     )
    #
    #     # generate object_detection files one by one
    #     # one image, one bbox_list; if no object_detection for an image, then bbox_list is empty
    #     prediction = load_json_detection_to_detection_list(json_file, test_image_list)
    #
    #     # conduct non max suppression
    #     pred = non_max_suppression(prediction, thd=0.5)
    #     video_detection_list.extend(pred)
    #     # save the object_detection to txt files
    #     if result_root_dir is not None:
    #         result_dir = os.path.join(result_root_dir, dir_name)
    #         smrc.line.generate_dir_if_not_exist(result_dir)
    #         print(f'Saving object_detection results to {result_dir} ... ')
    #         generate_detection_txt_with_image_list(pred, result_dir, test_image_list)
    #
    # return







def load_json_detection_to_detection_list(json_detection_file, image_list):
    """
    We did not call it bbox_list, because I used bbox_list to represents the object_detection format
    that without scores.
    bbox_list format [
        [class_idx, xmin, ymin, xmax, ymax],
        ...
    ]
    detection_list format [
        [class_idx, score, xmin, ymin, xmax, ymax],
        ...
    ]
    """
    detection_dict = load_json_detection_to_dict(json_detection_file)
    detection_list = det_dict_to_tracking_det_list (detection_dict, image_list)
    return detection_list

















def generate_detection_txt_with_image_list(result_dir, video_detection_list, test_image_list):
    assert len(test_image_list) == len(video_detection_list)

    # # generate txt object_detection file
    for image_path, detection_list in zip(test_image_list, video_detection_list):
        file_name = smrc.utils.image_path_last_two_level(image_path)
        detection_file_path = smrc.utils.replace_ext_str(file_name, 'txt')
        save_detection_to_file(detection_file_path, detection_list)

def generate_detection_txt_with_image_list(result_dir, video_detection_list, test_image_list):
    assert len(test_image_list) == len(video_detection_list)

    # # generate txt object_detection file
    for image_path, detection_list in zip(test_image_list, video_detection_list):
        file_name = smrc.utils.image_path_last_two_level(image_path)
        detection_file_path = smrc.utils.replace_ext_str(file_name, 'txt')
        save_detection_to_file(detection_file_path, detection_list)


# def generate_detection_or_gruth_txt_rename(detection_list):

#     assert len(detection_list) == len(ground_truth_list)
        
#     # # generate txt object_detection file
#     for idx in range(len(detection_list)):
#         object_detection = detection_list[idx]
#         detection_file_path = os.path.join('datasets', 'detections', format(idx, '06d') + '.txt')
#         save_detection_to_file(detection_file_path, object_detection)

#         # generate txt file for ground truth
#         truth = ground_truth_list[idx]
#         truth_file_path = os.path.join('datasets', 'groundtruths', format(idx, '06d') + '.txt')
#         smrc.line.save_bbox_to_file(truth_file_path, truth)




def save_detection_to_file(ann_path, detection_list):
    """
    detection_list, the detections for a single image of the format of 
    [
        [class_idx, score, xmin, ymin, xmax, ymax],
        ...
    ]
    """
    # save the bbox if it is not the active bbox (future version, active_bbox_idxs includes more than one idx)
    with open(ann_path, 'w') as new_file:
        for detection in detection_list:
            class_idx, score, xmin, ymin, xmax, ymax = detection
            items = map(str, [int(class_idx), score, round(xmin), round(ymin), \
                                round(xmax), round(ymax)])
            txt_line = ' '.join(items)
            # we need to add '\n'(newline), otherwise, all the bboxes will be in one line and not able to be recognized.
            new_file.write(txt_line + '\n')
    new_file.close()


# def generate_txt_file_for_detection(detection_list):
#
#     # # generate txt object_detection file
#     for idx in range(len(detection_list)):
#         object_detection = detection_list[idx]
#         detection_file_path = os.path.join('datasets', 'detections', format(idx, '06d') + '.txt')
#         save_detection_to_file(detection_file_path, object_detection)


def generate_txt_file_for_public_map_code(detection_list, ground_truth_list):

    assert len(detection_list) == len(ground_truth_list)
        
    # # generate txt object_detection file
    for idx in range(len(detection_list)):
        detection = detection_list[idx]
        detection_file_path = os.path.join('datasets', 'detections', format(idx, '06d') + '.txt')
        save_detection_to_file(detection_file_path, detection)

        # generate txt file for ground truth
        truth = ground_truth_list[idx]
        truth_file_path = os.path.join('datasets', 'groundtruths', format(idx, '06d') + '.txt')
        smrc.utils.save_bbox_to_file(truth_file_path, truth)




def non_max_suppression(raw_prediction, thd=0.5):
    
    print(f'Conducting non max suppression for detections of {len(raw_prediction)} files with thd {thd} ...')
    pred = []
    for image_pred in raw_prediction:
        if len(image_pred) == 0:
            bbox_processed = []
        else:
            raw_bbox_array = np.array(image_pred)
            boxes, scores = raw_bbox_array[:, 2:], raw_bbox_array[:, 1].flatten()

            if len(image_pred) > 1:
                selected = smrc.utils.non_max_suppression(boxes, scores, thd)
                # bbox_processed = raw_bbox_array[selected,0:6].astype("int").tolist()
                bbox_processed = raw_bbox_array[selected, 0:6].tolist()

                # if len(selected) < len(boxes): # selected, a list
                #     print(f'raw = {list(range(len(raw_bbox_array)))}')
                #     print(f'selected = {selected}')
                #     sys.exit(0)
            else:
                # bbox_processed = raw_bbox_array[:,0:6].astype("int").tolist()
                bbox_processed = raw_bbox_array[:, 0:6].tolist()

        pred.append(bbox_processed)
    print(f'Non max suppression done ...')
    return pred


def load_detection_txt_files_based_on_image_list(label_dir, test_image_path):
    assert len(test_image_path) > 0

    # each image corresponds to one list in prediction
    prediction_all = []
    for idx, image_path in enumerate(test_image_path):
        file_names = image_path.split(os.path.sep)
        pos = image_path.find(file_names[-2])
        image_dir = image_path[:pos-1]

        ann_path = smrc.utils.get_image_or_annotation_path(image_path, image_dir,
                                                           label_dir, '.txt')
        if not os.path.isfile(ann_path):
            detection = []
        else:
            detection = load_detection_from_file(ann_path)

        prediction_all.append(detection)

        # # generate txt object_detection file
        # detection_file_path = os.path.join('datasets', 'detections', format(idx, '06d') + '.txt')
        # save_detection_to_file(detection_file_path, object_detection)

    return prediction_all


# def load_prediction_from_public_mAP_format(label_dir):
#     exception_prediction_id = [2439, 2440, 6566, 6569] # 1-index
#     file_idx_list = []
#     # 2438, 2439.txt
#     num_non_empty = 0

#     ann_path_list = smrc.line.get_file_list_in_directory(label_dir)
#     prediction_all = []
#     for idx, ann_path in enumerate(ann_path_list):
#         object_detection = load_detection_normal_format_from_file(ann_path)
#         # each file corresponding to one element in prediction_all
#         prediction_all.append(object_detection)

        
#         num_non_empty += len(object_detection)
#         print(f'idx = {idx}, ann_path = {ann_path}, prediction {num_non_empty}')
#         if num_non_empty in exception_prediction_id:
#             print('=================================')
#         # if len(object_detection) > 0:
#         #     file_idx_list.extend( [idx] * len(object_detection))
        
#     # print(f'file_idx_list = {file_idx_list}')
#     # print(f'len(file_idx_list) = {len(file_idx_list)}')
#     # for pred_idx in exception_prediction_id:
#     #     tmp = file_idx_list[pred_idx-1]
#     #     print(f'{pred_idx}th prediction, ann_path = {ann_path_list[tmp]}')
#     sys.exit(0)
#     return prediction_all








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







def load_test_image_list(filename):
    return smrc.utils.load_1d_list_from_file(filename)


def generate_new_images(image_list, old_image_dir, new_image_dir):

    for idx, image_path in enumerate(image_list):  # detection_idx,
        # load the image name
        file_names = image_path.split(os.path.sep)
        pos = image_path.find(file_names[-2])
        image_dir = image_path[:pos - 1]

        source_image_path = smrc.utils.get_image_or_annotation_path(
            image_path, image_dir, old_image_dir, '.jpg')

        new_image_path = os.path.join(new_image_dir, format(idx, '06d') + '.jpg')
        # save_detection_to_file(detection_file_path, object_detection)
        print(f'Copying {image_path} ({idx+1}/{len(image_list)}) to {new_image_path} ...')
        shutil.copy(source_image_path, new_image_path)


