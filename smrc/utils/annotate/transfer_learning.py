import os
import random
import shutil
import sys
import cv2
from tqdm import tqdm
import numpy as np

import smrc.utils
from ..base import *
from ..image_video import get_image_file_list_in_directory


def save_image_list_to_file_incrementally(file_path, list_to_save):
    with open(file_path, 'a') as f:
        for item in list_to_save:
            f.write("%s\n" % item)


def shuffle_training_data_list(training_file_list):
    print(training_file_list)
    image_list = []
    for training_file in training_file_list:
        print(f'Starting to load {training_file}')
        training_data = smrc.utils.load_1d_list_from_file(training_file)
        # print(training_data)
        image_list.extend(training_data)
    random.shuffle(image_list)
    return image_list


def merge_and_shuffle_training_list(training_data_file_list, result_file_name):
    """
    # training_data_file_list = [
    #         'datasets/TruckDB467videos_non_empty_images86019.txt',
    #         'datasets/TruckDB/TruckDB159videos_non_empty_images25350.txt'
    # ]
    :param training_data_file_list:
    :param result_file_name:
    :return:
    """
    training_data = shuffle_training_data_list(
        training_data_file_list
    )
    smrc.utils.save_1d_list_to_file(result_file_name, training_data)


def load_bbox_from_txt_file(ann_path, label_format='smrc'):
    assert label_format in ['smrc', 'yolo']
    if label_format == 'smrc':
        # get bbox_list
        return smrc.utils.load_bbox_from_file(os.path.abspath(ann_path))
    elif label_format == 'yolo':
        return smrc.utils.annotate.load_yolo_bbox_from_file(
            os.path.abspath(ann_path), delimiter=' '
        )



def generate_training_data_image_list_file(
        image_dir, label_dir, label_format='smrc',
        video_list=None, experiment_flag='training_data', shuffle_enabled=True
):
    if video_list is None:
        dir_list = smrc.utils.get_dir_list_in_directory(label_dir)
    else:
        dir_list = video_list

    resulting_txt_root_dir = os.path.join(os.path.dirname(label_dir), 'transfer_learning_data')
    smrc.utils.generate_dir_if_not_exist(resulting_txt_root_dir)

    valid_training_data_flags = ['all', 'non_empty_images', 'non_empty_videos', 'all_existing_annotation']
    file_list_all = os.path.join(resulting_txt_root_dir, experiment_flag + '_all')
    file_list_non_empty_image = os.path.join(resulting_txt_root_dir, experiment_flag + '_non_empty_images')
    file_list_non_empty_video = os.path.join(resulting_txt_root_dir, experiment_flag + '_non_empty_videos')
    file_list_existing_annotation_txt = os.path.join(
        resulting_txt_root_dir, experiment_flag + '_all_existing_annotation'
    )

    file_list = [file_list_all, file_list_non_empty_image, file_list_non_empty_video,
                 file_list_existing_annotation_txt]
    for file_name in file_list:
        # file_name = os.path.join(label_dir, training_data_flag)
        open(file_name, 'w').close()
        # file_list.append(file_name)

    training_directory_list = dir_list
    print('To load {} directories.'.format(len(training_directory_list)))

    training_data_image_list = []
    for dir_idx, dir_name in enumerate(training_directory_list):
        dir_path = os.path.join(label_dir, dir_name)
        print('Processing directory: {}/{}, {}'.format(str(dir_idx + 1),
                                                       str(len(training_directory_list)), dir_path))
        # only load the last name
        image_dir_name = dir_path.replace(label_dir, image_dir, 1)
        # print()
        # print(f'image_dir_name = {image_dir_name}')
        image_path_list = smrc.utils.get_file_list_in_directory(
            image_dir_name, only_local_name=False
        )
        txt_path_list = smrc.utils.get_file_list_in_directory(
            image_dir_name, only_local_name=False, ext_str='.txt'
        )
        image_path_list = smrc.utils.exclude_one_list_from_another(
            image_path_list, txt_path_list
        )
        # print(file_list)

        video_image_list = []
        video_non_empty_image_list = []
        video_existing_annotation_image_list = []
        for file_idx, image_path in enumerate(image_path_list):
            # print(f'Transfering {filename}, {file_id}/{len(file_list)}')
            ann_path = smrc.utils.get_image_or_annotation_path(image_path, image_dir, label_dir, '.txt')

            # only handle the ann_path has corresponding images
            if os.path.isfile(image_path):  # cv2.imread(image_path) is not None:
                abs_image_path = os.path.abspath(image_path)
                video_image_list.append(abs_image_path)

                # for the images without a ann_txt, generate an empty file.
                if os.path.isfile(ann_path):
                    video_existing_annotation_image_list.append(abs_image_path)

                    # for iamges with txt file but have empty annotation, generate empty file
                    bbox_list = load_bbox_from_txt_file(label_format)
                    if len(bbox_list) > 0:
                        video_non_empty_image_list.append(abs_image_path)
            else:
                print(f'Image {image_path} does not exist, please check.')
                # os.remove(ann_path)
                # print(' File {} has been deleted'.format(ann_path))

        # if training_directory_flag == 'non_empty_images':
        #     training_data_image_list.extend(video_non_empty_image_list)
        # elif training_directory_flag == 'non_empty_videos':
        #     if len(video_non_empty_image_list) > 0:  # this video has annotated bbox
        #         training_data_image_list.extend(video_image_list)
        # elif training_directory_flag == 'all_existing_annotation':
        #     training_data_image_list.extend(video_existing_annotation_image_list)

        save_image_list_to_file_incrementally(file_list_all, video_image_list)
        save_image_list_to_file_incrementally(file_list_non_empty_image, video_non_empty_image_list)
        if len(video_non_empty_image_list) > 0:  # this video has annotated bbox
            save_image_list_to_file_incrementally(file_list_non_empty_video, video_image_list)

        save_image_list_to_file_incrementally(file_list_existing_annotation_txt,
                                              video_existing_annotation_image_list)

    # file_list_all = os.path.join(label_dir, 'all')
    # file_list_non_empty_image = os.path.join(label_dir, 'non_empty_images')
    # file_list_non_empty_video = os.path.join(label_dir, 'non_empty_videos')
    # file_list_existing_annotation_txt = os.path.join(label_dir, 'all_existing_annotation')

    if shuffle_enabled:
        for idx, training_data_flag in enumerate(valid_training_data_flags):
            file_name = file_list[idx]
            training_image_list = smrc.utils.load_1d_list_from_file(file_name)
            random.shuffle(training_image_list)

            resulting_file = os.path.join(
                resulting_txt_root_dir, experiment_flag + '_' + str(len(dir_list)) + 'videos_'
                           + training_data_flag + str(len(training_image_list)) + '.txt')

            smrc.utils.save_1d_list_to_file(resulting_file, training_image_list)

    # remove the intermediate files
    for file in file_list:
        os.remove(file)
        print('File {} has been deleted'.format(file))

# def generate_eval_data(
#         image_dir, label_dir, result_root_dir, max_image_num=5000,
#         video_list=None, experiment_flag='eval_data'
# ):
#     if video_list is None:
#         dir_list = smrc.line.get_dir_list_in_directory(label_dir)
#     else:
#         dir_list = video_list
#     print(f'To load {len(dir_list)} directories.')
#
#     smrc.line.generate_dir_if_not_exist(result_root_dir)
#     file_list_non_empty_image = os.path.join(result_root_dir, experiment_flag + '_non_empty_images')
#     open(file_list_non_empty_image, 'w').close()
#
#     for dir_idx, dir_name in enumerate(dir_list):
#         ann_dir_path = os.path.join(label_dir, dir_name)
#         print('Processing directory: {}/{}, {}'.format(str(dir_idx + 1),
#                                                        str(len(dir_list)), ann_dir_path))
#         video_non_empty_image_list = []
#         ann_path_list = smrc.line.get_file_list_in_directory(ann_dir_path)
#         for file_idx, ann_path in enumerate(ann_path_list):
#             bbox_list = smrc.line.load_bbox_from_file(os.path.abspath(ann_path))
#
#             image_path = smrc.line.get_image_or_annotation_path(ann_path, label_dir, image_dir, '.jpg')
#             if len(bbox_list) > 0:
#                 if os.path.isfile(image_path):
#                     video_non_empty_image_list.append(image_path)  # os.path.abspath(image_path)
#                 else:
#                     print(f'Image {image_path} does not exist, please check.')
#                     sys.exit(0)
#                     # os.remove(ann_path)
#                     # print(' File {} has been deleted'.format(ann_path))
#         save_image_list_to_file_incrementally(file_list_non_empty_image, video_non_empty_image_list)
#
#     training_image_list = smrc.line.load_1d_list_from_file(file_list_non_empty_image)
#     random.shuffle(training_image_list)
#
#     if len(training_image_list) > max_image_num:
#         training_image_list = training_image_list[:max_image_num]
#
#     resulting_image_dir = os.path.join(result_root_dir, experiment_flag + '_images')
#     resulting_label_dir = os.path.join(result_root_dir, experiment_flag + '_labels')
#
#     smrc.line.generate_dir_if_not_exist(resulting_image_dir)
#     smrc.line.generate_dir_if_not_exist(resulting_label_dir)
#
#     image_path_list_final = []
#     for i, image_path in enumerate(training_image_list):
#         ann_path = smrc.line.get_image_or_annotation_path(image_path, image_dir, label_dir, '.txt')
#         new_image_path = os.path.join(resulting_image_dir, image_path.replace(os.path.sep, '_'))
#
#         new_ann_path = smrc.line.get_image_or_annotation_path(
#             new_image_path,
#             resulting_image_dir,
#             resulting_label_dir, '.txt'
#         )
#         print(f'Copying {i}th/{len(training_image_list)} image to {new_image_path}, so as ann file ... ')
#         shutil.copy(image_path, new_image_path)
#         shutil.copy(ann_path, new_ann_path)
#
#         image_path_list_final.append(os.path.abspath(new_image_path))
#
#         # visualization
#         tmp_img = cv2.imread(image_path)
#         AnnotationTool.draw_bboxes_from_file(tmp_img, ann_path)
#         result_image_filename = os.path.join(resulting_visualization_dir, image_path.replace(os.path.sep, '_'))
#         cv2.imwrite(result_image_filename, tmp_img)
#
#     resulting_file = os.path.join(
#         result_root_dir,
#         experiment_flag + '_' + str(len(dir_list)) + 'videos_non_empty_images' + str(len(training_image_list)) + '.txt'
#     )
#     smrc.line.save_1d_list_to_file(resulting_file, image_path_list_final)


class TransferLearning:
    # def __init__:

    def balanced_data(
            self, image_root_dir, label_root_dir, result_file_name_prefix,
            label_format='yolo', dir_list=None
    ):
        if dir_list is None:
            dir_list = smrc.utils.get_dir_list_in_directory(label_root_dir)
        print(f'To process {len(dir_list)} directories for {label_root_dir}, format {label_format}')

        # generate the annotation dict, key: image path, value: a list of existing class id
        ann_dict = {}
        # statistic
        statistic_dict = {}

        pbar = tqdm(enumerate(dir_list))
        for dir_idx, dir_name in pbar:
            pbar.set_description(
                f'To counting labels in {label_format}: {dir_name} [{dir_idx+1}/{len(dir_list)}] ...')
            label_dir = os.path.join(label_root_dir, dir_name)
            if not os.path.isdir(label_dir):
                # len(get_file_list_in_directory(label_dir)) == 0
                continue

            # load the txt file (annotation file) under the directory
            image_path_list = get_image_file_list_in_directory(
                os.path.join(image_root_dir, dir_name)
            )

            # we check txt file from image file instead of the inverse order, because the format
            # of images may not determined (e.g., jpg, png, ...), while the annotation format is
            # fixed, .txt format
            for file_idx, image_path in enumerate(image_path_list):
                ann_path = get_image_or_annotation_path(
                    image_path, image_root_dir, label_root_dir, '.txt'
                )
                # skip when there is no corresponding annotation file
                if not os.path.isfile(ann_path): continue

                bbox_list = load_bbox_from_txt_file(ann_path, label_format=label_format)
                if len(bbox_list) > 0:
                    # extract only the class ids
                    class_id_list = [x[0] for x in bbox_list]
                    ann_dict[os.path.abspath(image_path)] = class_id_list
                    # update the count
                    for class_id in class_id_list:
                        if class_id not in statistic_dict:
                            statistic_dict[class_id] = 1
                        else:
                            statistic_dict[class_id] += 1

        sorted_class_ids = self._display_sorted_statistics(statistic_dict)
        # sample from the minority classes to the majority classes and save to files
        per_class_statistic_dict = {}
        per_class_ann_dict = {}
        for class_id in sorted_class_ids:
            print(f'Begin to extract image list for class {class_id} ... ')
            result_file_name = result_file_name_prefix + '_' + str(class_id)
            self.generate_empty_txt_file(result_file_name)
            keys = list(ann_dict.keys())
            for image_path in keys:
                class_id_list = ann_dict[image_path]
                if class_id in class_id_list:
                    per_class_ann_dict[image_path] = ann_dict[image_path]
                    del ann_dict[image_path]

                    for k in class_id_list:
                        if k not in per_class_statistic_dict:
                            per_class_statistic_dict[k] = 1
                        else:
                            per_class_statistic_dict[k] += 1
            image_path_list = list(per_class_ann_dict.keys())
            random.shuffle(image_path_list)
            save_1d_list_to_file(result_file_name, image_path_list)
            self._display_sorted_statistics(per_class_statistic_dict)

            statistics_result_file_name = result_file_name_prefix + '_' + str(class_id) \
                                          + '_' + 'statistics'
            statistics_list_to_save = [[k, v] for k, v in sorted(
                per_class_statistic_dict.items(), key=lambda item: item[0])]
            save_multi_dimension_list_to_file(
                filename=statistics_result_file_name,
                list_to_save=statistics_list_to_save,
                delimiter=' '
            )

    @staticmethod
    def generate_empty_txt_file(result_file_name):
        open(result_file_name, 'w').close()

    @staticmethod
    def _display_sorted_statistics(statistic_dict):
        """display the statistics
        :param statistic_dict: key, class id, value, class_id_list
        :return:
        """
        sorted_class_ids = [k for k, v in sorted(statistic_dict.items(), key=lambda item: item[1])]
        print('========================================================== ')
        total = np.sum([v for _, v in statistic_dict.items()])
        for class_id in sorted_class_ids:
            print(f'class id {class_id}, count {statistic_dict[class_id]}, '
                  f'{"%2.2f" % (float(statistic_dict[class_id]) * 100 / total)}%')
        return sorted_class_ids
