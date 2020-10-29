from .base import *
from .bbox import load_bbox_from_file
import os
from tqdm import tqdm


def count_bbox_number(label_dir, result_file_name=None):
    """
    # label_dir = 'Truck_SampleData_Facelabels'
    # label_dir = 'Truck_SampleData_licensePlatelabels_fitted'
    label_dir = 'test_data/DB_Image_out_labels-thd0.05'
    # 車内image
    :param label_dir:
    :param result_file_name: where to save the results
    :return:
    """
    dir_list = get_dir_list_in_directory(label_dir)
    count_list = []
    for dir_name in dir_list:

        ann_path_list = get_file_list_recursively(
            os.path.join(label_dir,dir_name)
        )

        bbox_num = 0
        for ann_path in ann_path_list:
            bbox_list = load_bbox_from_file(ann_path)
            bbox_num += len(bbox_list)
        count_list.append([dir_name, bbox_num])

    item_count_sorted_index = sorted(
        range(len(count_list)),
        key=lambda k: count_list[k][1],
        reverse=True
    )  # sort the list from large to small

    result = []
    for idx in item_count_sorted_index:
        dir_name, bbox_num = count_list[idx]
        print(f'{dir_name}\t {bbox_num}')
        result.append([dir_name, bbox_num])

    if result_file_name is None:
        result_file_name = 'count_bbox_result_' + time_stamp_str()

    save_multi_dimension_list_to_file(
        filename=result_file_name, list_to_save=result,
        delimiter=','
    )

    return result


def report_train_test_statistics(smrc_label_dir, train_video_list, test_video_list):
    print(f'All the annotated data ... ')
    estimate_annotation_statistics(
        smrc_label_dir
    )
    print(f'Training data ... ')
    estimate_annotation_statistics(
        smrc_label_dir, train_video_list
    )
    print(f'Test data ... ')
    estimate_annotation_statistics(
        smrc_label_dir, test_video_list
    )


def estimate_and_report_statistics(root_dir, class_list_name=None):
    """
    # report the statistics for the annotation
    annotation_root_dir = 'TruckDB/TEST-50videos-labels'
    class_list_name = os.path.join('config/class.names')

    :param root_dir:
    :param class_list_name:
    :return:
    """
    num_of_bbox_per_class, class_ids, num_of_txt_file, num_of_bboxes, num_of_non_empty_image = \
        estimate_annotation_statistics(root_dir)

    if class_list_name is None:
        class_list = ['obj-' + str(x) for x in class_ids]
    else:
        # load the class list
        class_list = load_1d_list_from_file(class_list_name)
    print('=======================================================')
    print(f'num_of_txt_file = {num_of_txt_file}, '
          f'num_of_non_empty_image ={num_of_non_empty_image}, '
          f'num_of_bbox = {num_of_bboxes}, num_of_class ={len(class_ids)}')

    # for latex or excel use
    for class_idx in class_ids:
        percent_str = format(float(num_of_bbox_per_class[class_idx]) * 100 / num_of_bboxes, '4.1f')
        print(f'{str(class_idx)}\t{class_list[class_idx]}\t{str(num_of_bbox_per_class[class_idx])}\t{percent_str}')
        # print(f'{str(class_idx).ljust(2)}\t{class_list[class_idx].ljust(10)}\t{str(num_of_bbox_per_class[class_idx]).ljust(8)}\t{percent_str}%')
    # print('=======================================================')
    # print('Github use')
    # # for github use
    # max_class_name_length = 0
    # for class_name in class_list:
    #     if len(class_name) > max_class_name_length:
    #         max_class_name_length = len(class_name)
    #
    # for class_idx in class_ids:
    #     percent_str = format(float(num_of_bbox_per_class[class_idx]) * 100 / num_of_bboxes, '4.1f')
    #
    #     print(f'{str(class_idx).ljust(2)}\t{class_list[class_idx].ljust(max_class_name_length)}\t{str(num_of_bbox_per_class[class_idx]).ljust(8)}\t{percent_str}%')
    # print('=======================================================')


def estimate_annotation_statistics(root_dir, dir_list=None):
    if dir_list is None:
        dir_list = get_dir_list_recursively(root_dir)
    else:
        dir_list = [os.path.join(root_dir, dir_name) for dir_name in dir_list]
    statistics = get_statistics_for_annotation_results(dir_list)
    return statistics


def get_statistics_for_annotation_results(dir_list):
    num_of_txt_file, num_of_non_empty_image, num_of_bboxes = 0, 0, 0
    class_ids = []
    num_of_bbox_per_class = {}

    for dir_idx, dir_path in tqdm(enumerate(dir_list)):
        # print('Checking directory:  {}, {}/{}'.format(dir_path, str(dir_idx + 1),
        #                                               str(len(dir_list))))

        # load the txt file (annotation file) under the directory
        txt_file_list = get_file_list_in_directory(
            dir_path, only_local_name=False
        )

        num_of_txt_file += len(txt_file_list)

        for file_idx, file_name in enumerate(txt_file_list):
            annotated_bboxes = load_bbox_from_file(file_name)
            num_of_bboxes += len(annotated_bboxes)
            if len(annotated_bboxes) > 0: num_of_non_empty_image += 1

            for bbox_idx, bbox in enumerate(annotated_bboxes):
                class_idx, xmin, ymin, xmax, ymax = bbox

                if class_idx not in class_ids:
                    class_ids.append(class_idx)
                    num_of_bbox_per_class[class_idx] = 1
                else:
                    num_of_bbox_per_class[class_idx] += 1

                    # if class_idx <= 5:

                # else:
                # print('directory: {}, file_name: {}, bbox_idx : {}'.format(dir_path, file_name,
                # str(bbox_idx))

    print(f'num_of_txt_file = {num_of_txt_file}, '
          f'num_of_non_empty_image ={num_of_non_empty_image}, '
          f'num_of_bbox = {num_of_bboxes}, num_of_class ={len(class_ids)}')

    class_ids.sort()
    for class_idx in class_ids:
        print('class   %d    %d  %4.1f' %
              (class_idx, num_of_bbox_per_class[class_idx],
               float(num_of_bbox_per_class[class_idx]) * 100 / num_of_bboxes)
              )

    return num_of_bbox_per_class, class_ids, num_of_txt_file, num_of_bboxes, num_of_non_empty_image


def get_txt_file_list_for_specific_annotated_classes(root_dir, specified_class_IDs):
    # specified_class_IDs: a list, e.g., [1, 2, 3]

    num_of_bbox_found = 0
    resulting_txt_file_list = []

    # load the txt file (annotation file) under the directory
    txt_file_list = get_file_list_recursively(root_dir)
    for file_idx, file_name in enumerate(txt_file_list):
        print(f'Checking file:  {file_name}, {file_idx + 1}/{len(txt_file_list)}')
        bbox_list = load_bbox_from_file(file_name)

        for bbox in bbox_list:
            class_idx = bbox[0]

            if class_idx in specified_class_IDs:
                num_of_bbox_found += 1
                # print(f'resulting_txt_file_list = {resulting_txt_file_list}')
                if file_name not in resulting_txt_file_list:
                    resulting_txt_file_list.append(file_name)
                    # print(f'resulting_txt_file_list = {resulting_txt_file_list}')

    print(f'{num_of_bbox_found} found in {specified_class_IDs}, {len(resulting_txt_file_list)} txt files are related.')
    return resulting_txt_file_list
