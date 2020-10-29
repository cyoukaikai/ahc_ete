import os
import cv2
import re
import json
import numpy as np
import sys
import random
import shutil
import smrc.utils


class AnnotationPostProcess:
    def __init__(self,
                 image_dir,
                 label_dir_to_check,
                 checked_result_dir=None,
                 class_list_file=None,
                 directory_list_to_check=None,
                 operation='check'  #correct
                 ):

        self.image_dir = image_dir
        self.label_dir_to_check = label_dir_to_check

        if checked_result_dir is None:
            self.checked_result_dir = self.label_dir_to_check + '_tmp_SMRC_FORMAT'
        else:
            self.checked_result_dir = checked_result_dir

        self.class_index_list = []
        if class_list_file is not None:
            class_list = smrc.utils.load_class_list_from_file(class_list_file)
        # assert len(class_list) > 0
            self.class_index_list = list(range(len(class_list)))

        print('====================================================')
        print(f'To check and correct annotation results in {self.label_dir_to_check}.')
        print(f'The images in {self.image_dir} will be used to assist the checking.')
        print(f'The corrected results will be saved in {self.checked_result_dir}.')
        if len(self.class_index_list) > 0:
            print(f'class_list = [{len(self.class_index_list)}]: {self.class_index_list} ... ')
        print('====================================================')

        if operation.lower() == 'correct':
            self.check_and_correct_annotation_results(
                self.label_dir_to_check,
                self.checked_result_dir,
                directory_list_to_check
            )
        else:
            self.check_annotated_bbox(
                self.label_dir_to_check,
                directory_list_to_check
            )

    def check_and_correct_annotation_results(self, label_dir_to_check, checked_result_dir, directory_list=None):
        """Import the annotation results and conduct checking and correcting operations
        # input:
        #   the annotation results in [label_dir],
        #   with the assistance of the images in [image_dir]
        # output:
        #   the checked results [checked_result_dir]
        :param label_dir_to_check:
        :param checked_result_dir:
        :param directory_list:
        :return:
        """
        if directory_list is None:
            directory_list = smrc.utils.get_dir_list_in_directory(label_dir_to_check)

        assert len(directory_list) > 0, f'The annotation results in ' \
            f'{label_dir_to_check} do not exist. Please check ...'

        print('To check and correct annotation_results for {} directories...'.format(len(directory_list)))
        print('====================================================')
        smrc.utils.generate_dir_if_not_exist(checked_result_dir)

        # check if the annotations are valid
        for dir_idx, dir_name in enumerate(directory_list):
            dir_path = os.path.join(label_dir_to_check, dir_name)
            print('Processing {}, {}/{}'.format(dir_path, str(dir_idx + 1),
                                                         str(len(directory_list))))
            ann_dir_smrc_format = os.path.join(checked_result_dir, dir_name)
            smrc.utils.generate_dir_if_not_exist(ann_dir_smrc_format)

            # load the txt file (annotation file) under the directory
            txt_file_list = smrc.utils.get_file_list_in_directory(dir_path)

            # for each txt file, load the image, get the image_width and image_height
            # check if the annotated bboxes are valid.
            for file_idx, file_name in enumerate(txt_file_list):
                annotated_bboxes = smrc.utils.load_bbox_from_file(file_name)
                # print(annotated_bboxes)

                image_path = self.get_image_path_from_annotation_path(file_name)

                # check if it is an image
                test_img = cv2.imread(image_path)

                bbox_processed = []
                if test_img is not None:
                    image_height, image_width = test_img.shape[:2]

                    for bbox_idx, bbox in enumerate(annotated_bboxes):
                        class_idx, xmin, ymin, xmax, ymax = bbox

                        if len(self.class_index_list) > 0 and not self.is_valid_class_idx(class_idx):
                            print(f'The class {class_idx} for [{bbox}] '
                                  f'is not in class_list {self.class_index_list}, deleted ...')
                            continue
                        elif xmin == xmax or ymin == ymax:
                            print(f'xmin = {xmin}, xmax = {xmax}, ymin = {ymin}, ymax = {ymax}, deleted ...')
                            continue
                        else:  # valid annotation
                            if not smrc.utils.is_valid_bbox_rect(xmin, ymin, xmax, ymax, image_width, image_height):
                                print(f' Annotation {bbox} invalid: image_height = {image_height}, '
                                      f'image_width = {image_width}'
                                )
                                xmin, ymin, xmax, ymax = smrc.utils.post_process_bbox_coordinate(
                                    xmin, ymin, xmax, ymax, image_width, image_height
                                )
                                bbox = [class_idx, xmin, ymin, xmax, ymax]
                            # else:
                            # print('     Bbox {} is valid: xmin = {}, ymin = {}, xmax = {}, ymax = {}'.format(
                            # str(bbox_idx), str(bbox[1]), str(bbox[2]), str(bbox[3]), str(bbox[4]))
                            # )
                            bbox_processed.append(bbox)
                else:
                    print(' Can not load image {}, skip processing {} ...'.format(image_path, file_name))
                    continue

                    # Note it is dangerous or even a disaster to remove the annotation without the
                    # agreement of the user. So never run 'os.remove(file_name)'
                    # print(' File {} has been deleted'.format(file_name))

                ann_file_smrc_format = file_name.replace(label_dir_to_check, checked_result_dir, 1)
                smrc.utils.save_bbox_to_file(ann_file_smrc_format, bbox_processed)

    def check_annotated_bbox(self, label_dir_to_check, directory_list=None):
        if directory_list is None:
            directory_list = smrc.utils.get_dir_list_in_directory(label_dir_to_check)

        assert len(directory_list) > 0, f'The annotation results in ' \
            f'{label_dir_to_check} do not exist. Please check ...'

        # check if the annotations are valid
        # transfer the data to Yolo format
        for dir_idx, dir_name in enumerate(directory_list):
            dir_path = os.path.join(label_dir_to_check, dir_name)

            pass_flag = True

            # load the txt file (annotation file) under the directory
            txt_file_list = smrc.utils.get_file_list_in_directory(dir_path)
            # for each txt file, load the image, get the image_width and image_height
            # check if the annotated bboxes are valid.
            # Print the bbox that are found invalid
            for file_idx, file_name in enumerate(txt_file_list):
                annotated_bboxes = smrc.utils.load_bbox_from_file(file_name)
                # print(annotated_bboxes)
                image_path = self.get_image_path_from_annotation_path(file_name)

                # check if it is an image
                test_img = cv2.imread(image_path)

                if test_img is not None:
                    image_height, image_width = test_img.shape[:2]
                    # print(' Load image {}, image_height = {}, image_width = {}'.format(
                    #     image_path, str(image_height), str(image_width)
                    # ))

                    for bbox_idx, bbox in enumerate(annotated_bboxes):
                        # class_idx, xmin, ymin, xmax, ymax = bbox
                        # print(xmin, ymin, xmax, ymax )
                        if not self.is_valid_bbox(bbox, image_width, image_height):
                            print('     Annotation {} is invalid: class_idx = {}, xmin = {}, '
                                  'ymin = {}, xmax = {}, ymax = {}, '
                                  'image_height = {}, image_width = {}'.format(
                                    file_name, str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3]), str(bbox[4]),
                                    str(image_height), str(image_width))
                            )
                            pass_flag = False
                else:
                    pass_flag = False
                    print(' Can not load image {}'.format(image_path))
            check_result = 'Passed' if pass_flag else 'Failed'
            print('{}: {}, {}/{}'.format(check_result, dir_path, str(dir_idx + 1), len(directory_list)))

    def is_valid_bbox(self, bbox, image_width, image_height):
        class_idx, xmin, ymin, xmax, ymax = bbox

        return (self.is_valid_class_idx(class_idx) and
                smrc.utils.is_valid_bbox_rect(xmin, ymin, xmax, ymax, image_width, image_height)
                )

    def is_valid_class_idx(self, class_idx):
        return class_idx in self.class_index_list

    def get_image_path_from_annotation_path(self, txt_file_name):
        return smrc.utils.get_image_or_annotation_path(
            txt_file_name,
            self.label_dir_to_check,
            self.image_dir, '.jpg'
        )