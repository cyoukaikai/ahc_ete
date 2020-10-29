#!/bin/python
import os
import cv2 
import smrc.utils


def load_yolo_bbox_from_file(ann_path, delimiter=' '):
    annotated_bbox = []
    if os.path.isfile(ann_path):
        with open(ann_path, 'r') as old_file:
            lines = old_file.readlines()
        old_file.close()

        # print('lines = ',lines)
        for line in lines:
            # txt format
            result = line.split(delimiter)

            bbox = [int(result[0]), float(result[1]), float(result[2]), float(
                result[3]), float(result[4])]
            annotated_bbox.append(bbox)

    return annotated_bbox


def bbox_transfer_to_yolo_format(class_index, point_1, point_2, image_width, image_height):
    # borrowed from OpenLabeling
    # YOLO wants everything normalized
    # print(point_1, point_2, image_width, image_height)
    # Order: class x_center y_center x_width y_height
    x_center = (point_1[0] + point_2[0]) / float(2.0 * image_width)
    y_center = (point_1[1] + point_2[1]) / float(2.0 * image_height)
    x_width = float(abs(point_2[0] - point_1[0])) / image_width
    y_height = float(abs(point_2[1] - point_1[1])) / image_height
    items = map(str, [int(class_index), x_center, y_center, x_width, y_height])
    return ' '.join(items)


def save_bbox_to_file_yolo_format(ann_path, bbox_list, image_width, image_height):
    # save the bbox if it is not the active bbox (future version, active_bbox_idxs includes more than one idx)
    with open(ann_path, 'w') as new_file:
        for idx, bbox in enumerate(bbox_list):
            class_idx, xmin, ymin, xmax, ymax = bbox
            txt_line = bbox_transfer_to_yolo_format(class_idx, (xmin, ymin), (xmax, ymax),
                                                    image_width, image_height)
            # we need to add '\n'(newline), otherwise, all the bboxes will be in one line and not able to be recognized.
            new_file.write(txt_line + '\n')
    new_file.close()


def transfer_smrc_label_to_yolo_format(
        image_dir, smrc_label_dir, yolo_format_dir,
        generate_empty_ann_file_flag=True,
        # video_flag=False,
        # check_image_existence=True,
        post_process_bbox_list=False,
        class_list=None,
        dir_list=None
):
    """
    :param image_dir:
    :param smrc_label_dir:
    :param yolo_format_dir:
    :param generate_empty_ann_file_flag: if true, generate a txt file for each image;
        otherwise, skip generating empty annotation file
    # :param video_flag: if video, then only load the first image to obtain the image size (height, width);
    #     otherwise, load every image to obtain the image size.
    :param dir_list: if not given, then load all the dirs in smrc_label_dir
    :return:
    """

    if dir_list is None:
        dir_list = smrc.utils.get_dir_list_in_directory(smrc_label_dir)

    assert len(dir_list) > 0, f'dir_list is empty, please check. '
    smrc.utils.generate_dir_if_not_exist(yolo_format_dir)

    for dir_index, dir_name in enumerate(dir_list):
        print(f'Generating YOLO format for {dir_name}, {dir_index + 1}/{len(dir_list)} ...')
        smrc.utils.generate_dir_if_not_exist(
            os.path.join(yolo_format_dir, dir_name)
        )

        image_dir_name = os.path.join(image_dir, dir_name)
        image_path_list = smrc.utils.get_file_list_recursively(image_dir_name)
        # if video_flag:
        #     height, width = smrc.line.get_image_size(image_path_list[0])
        #     print(f'video infor: height = {height}, width = {width} ...')
        # else:
        #     print(f'video infor: height = {height}, width = {width} ...')

        for file_id, image_path in enumerate(image_path_list):
            # print(f'Transfering {filename}, {file_id}/{len(file_list)}')
            ann_path = smrc.utils.get_image_or_annotation_path(image_path, image_dir, smrc_label_dir, '.txt')
            bbox_list = smrc.utils.load_bbox_from_file(ann_path)

            ann_path_new = ann_path.replace(smrc_label_dir, yolo_format_dir, 1)
            if len(bbox_list) == 0:
                if generate_empty_ann_file_flag:
                    smrc.utils.empty_annotation_file(ann_path_new)
                continue

            # if there are at least one bbox
            img = cv2.imread(image_path)
            if img is not None:
                height, width, _ = img.shape
                if post_process_bbox_list:
                    bbox_list = smrc.utils.post_process_bbox_list(bbox_list, height, width, class_list=class_list)
                save_bbox_to_file_yolo_format(
                    ann_path_new, bbox_list, width, height
                )
            else:
                print(f'Image {image_path} does not exist, please check.')
                # os.remove(ann_path)
                # print(' File {} has been deleted'.format(ann_path))


def generate_single_label_data_from_yolo_format(
        class_list, yolo_label_dir, dir_list=None,
        keep_class_label=False
):
    if dir_list is None:
        dir_list = os.listdir(yolo_label_dir)

    for class_id in class_list:
        result_dir = yolo_label_dir + '_' + str(class_id)
        smrc.utils.generate_dir_if_not_exist(result_dir)

    for idx, dir_name in enumerate(dir_list):
        print(f'Extracting data from {yolo_label_dir}/{dir_name}, {idx}/{len(dir_list)} ...')
        ann_file_list = smrc.utils.get_file_list_recursively(
            os.path.join(yolo_label_dir, dir_name)
        )

        for class_id in class_list:
            result_dir = yolo_label_dir + '_' + str(class_id)
            print(f'generating {os.path.join(result_dir, dir_name)} ...')
            smrc.utils.generate_dir_if_not_exist(
                os.path.join(result_dir, dir_name)
            )

        for ann_path in ann_file_list:
            bbox_list = load_yolo_bbox_from_file(ann_path)
            # if len(bbox_list) == 0:
            #     continue

            for class_id in class_list:

                if keep_class_label:
                    bbox_extracted = [x for x in bbox_list if x[0] == class_id]
                else:
                    # set all the class labels to be 0
                    bbox_extracted = [[0] + x[1:] for x in bbox_list if x[0] == class_id]
                # Do not generate txt file if there is no annotation
                if len(bbox_extracted) == 0:
                    continue

                result_dir = yolo_label_dir + '_' + str(class_id)
                ann_path_new = ann_path.replace(yolo_label_dir, result_dir, 1)
                smrc.utils.save_multi_dimension_list_to_file(
                    os.path.abspath(ann_path_new), bbox_extracted, delimiter=' '
                )


