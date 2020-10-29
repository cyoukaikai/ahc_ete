import shutil
from tqdm import tqdm
import os

# import utils
from .base import *
# get_file_list_in_directory, get_dir_list_in_directory, \
# get_image_or_annotation_path, diff_list, generate_dir_if_not_exist
from .image_video import get_image_file_list_in_directory, \
    get_image_size
from .bbox import empty_annotation_file
from .annotate.ann_utils import replace_dir_sep_with_underscore

import cv2


def find_dir_diff(root_dir1, root_dir2):
    dir_list1 = get_dir_list_in_directory(root_dir1, only_local_name=True)
    dir_list2 = get_dir_list_in_directory(root_dir2, only_local_name=True)
    unique_list1, unique_list2 = diff_list(dir_list1, dir_list2)
    return unique_list1 + unique_list2

#########################################################
# yolo-v4, data clean
############################################################


def remove_txt_files(image_root_dir):
    """
    :param image_root_dir:
    :return:
    """
    print(f'To remove all the txt files in {image_root_dir} ...')
    count = 0
    dir_list = get_dir_list_in_directory(image_root_dir)

    pbar = tqdm(enumerate(dir_list))
    for dir_idx, dir_name in pbar:
        pbar.set_description(f'To remove all the txt files in {image_root_dir}: '
                             f'Processing {dir_name} ({dir_idx}/{len(dir_list)}) ...')
        # print(f'Processing {dir_name} ({dir_idx}/{len(dir_list)}) ...')
        txt_file_list = get_file_list_in_directory(
            os.path.join(image_root_dir, dir_name), ext_str='.txt'
        )
        for txt_file in txt_file_list:
            os.remove(txt_file)
            count += 1
    print(f'Total {count} files removed ...')


def copy_labels(source_label_dir, target_label_dir):
    print(f'To copy labels from {source_label_dir} to {target_label_dir} ...')
    dir_list = get_dir_list_in_directory(source_label_dir)
    count = 0
    pbar = tqdm(enumerate(dir_list))
    for dir_idx, dir_name in pbar:
        pbar.set_description(f'To copy labels from {source_label_dir} to {target_label_dir} :'
                             f'{dir_name} ({dir_idx}/{len(dir_list)}) ...')
        # print(f'Processing {dir_name} ({dir_idx}/{len(dir_list)}) ...')
        txt_file_list = get_file_list_in_directory(
            os.path.join(source_label_dir, dir_name),
            ext_str='.txt', only_local_name=False
        )
        for txt_file in txt_file_list:
            target_file = txt_file.replace(
                source_label_dir, target_label_dir, 1
            )
            # print(f'Copying {txt_file} to {target_file} ...')
            shutil.copyfile(txt_file, target_file)
            count += 1
    print(f'Total {count} files copied ...')


def count_empty_txt_file(label_root_dir):
    """
    root_dir = os.path.join('datasets', 'DensoData_No_Masking8SampleVideo')
    parent_dir_to_process = os.path.join(root_dir, 'labels-first98videos') #tmp_YOLO_FORMAT3807

    label_root_dir = 'data/datasets/TruckDB_Face/all7594'
    :param label_root_dir:
    :return:
    """
    dir_list = get_dir_list_in_directory(label_root_dir)
    count = 0
    # dir_list = dir_list[0]
    pbar = tqdm(enumerate(dir_list))
    for dir_idx, dir_name in pbar:
        pbar.set_description(f'Processing {dir_name} ({dir_idx}/{len(dir_list)}) ...')
        # print(f'Processing {dir_name} ({dir_idx}/{len(dir_list)}) ...')
        txt_file_list = get_file_list_in_directory(
            os.path.join(label_root_dir, dir_name), ext_str='.txt'
        )
        for txt_file in txt_file_list:
            statinfo = os.stat(txt_file)
            # print(f'{txt_file}, {statinfo.st_size}')
            if statinfo.st_size == 0:
                print(f'{txt_file} empty ... ')
                count += 1
    print(f'Total {count} files empty ...')


def padding_empty_files(image_root_dir, label_root_dir=None):
    """Generating empty files for images without annotation (empty images).
    """
    # for yolo v4, the labels and images are in the same directory
    # for yolo v3, labels and images are in different directories.
    if label_root_dir is None:
        label_root_dir = image_root_dir
    print(f'To padding labels to {image_root_dir} based on txt files in {label_root_dir} ...')
    dir_list = get_dir_list_in_directory(image_root_dir)
    count = 0
    pbar = tqdm(enumerate(dir_list))
    for dir_idx, dir_name in pbar:
        pbar.set_description(f'To padding labels to {image_root_dir}: {dir_name} [{dir_idx}/{len(dir_list)}] ...')
        # print(f'Processing {dir_name} ({dir_idx}/{len(dir_list)}) ...')
        image_path_list = get_image_file_list_in_directory(
            os.path.join(image_root_dir, dir_name)
        )
        for image_path in image_path_list:
            txt_file_name = get_image_or_annotation_path(
                image_path, image_root_dir, label_root_dir, '.txt'
            )
            if not os.path.isfile(txt_file_name):
                empty_annotation_file(txt_file_name)
                # print(f'Generating empty file {txt_file_name} ...')
                count += 1
    print(f'Total {count} empty files generated ...')


def prepare_yolov4_label(
        image_root_dir, yolo_label_root_dir, padding_empty_file_flag=False
):
    # remove all the txt files
    remove_txt_files(image_root_dir)

    # copy the txt files to image dir
    copy_labels(
        source_label_dir=yolo_label_root_dir,
        target_label_dir=image_root_dir
    )
    # padding empty files
    if padding_empty_file_flag:
        padding_empty_files(image_root_dir=image_root_dir)


#########################################################
# resize images
#########################################################

def delete_empty_file(label_root_dir):
    """
    root_dir = os.path.join('datasets', 'DensoData_No_Masking8SampleVideo')
    parent_dir_to_process = os.path.join(root_dir, 'labels-first98videos') #tmp_YOLO_FORMAT3807

    label_root_dir = 'data/datasets/TruckDB_Face/all7594'
    :param label_root_dir:
    :return:
    """
    dir_list = get_dir_list_in_directory(label_root_dir)

    # dir_list = dir_list[0]
    for dir_idx, dir_name in enumerate(dir_list):
        print(f'Processing {dir_name} ({dir_idx}/{len(dir_list)}) ...')
        txt_file_list = get_file_list_in_directory(
            os.path.join(label_root_dir, dir_name)
        )
        for txt_file in txt_file_list:
            statinfo = os.stat(txt_file)
            print(f'{txt_file}, {statinfo.st_size}')
            if statinfo.st_size == 0:
                print(f'Removing {txt_file} ... ')
                os.remove(txt_file)

        txt_file_list_remains = get_file_list_in_directory(
            os.path.join(label_root_dir, dir_name)
        )
        print(f'After processing {dir_name}, {len(txt_file_list_remains)} of {len(txt_file_list)} remains ')


def resize_image_pairwise_comparison(src_image_root_dir, refer_image_root_dir):
    """
    Resize the images in src_image_root_dir based on the size of the corresponding images in refer_image_root_dir
    :param src_image_root_dir:
    :param refer_image_root_dir:
    :return:
    """
    assert len(find_dir_diff(src_image_root_dir, refer_image_root_dir)) == 0
    target_image_root_dir = src_image_root_dir + '_resized'
    generate_dir_if_not_exist(target_image_root_dir)
    print(f'To resize images in {src_image_root_dir} based on the corresponding images in '
          f'{refer_image_root_dir}, results are saved in  {target_image_root_dir} ...')

    dir_list = get_dir_list_in_directory(src_image_root_dir)
    count = 0
    pbar = tqdm(enumerate(dir_list))
    for dir_idx, dir_name in pbar:
        pbar.set_description(f'To process {dir_name} [{dir_idx}/{len(dir_list)}] ...')
        # print(f'Processing {dir_name} ({dir_idx}/{len(dir_list)}) ...')
        image_path_list = get_image_file_list_in_directory(
            os.path.join(src_image_root_dir, dir_name)
        )
        target_dir_path = os.path.join(target_image_root_dir, dir_name)
        generate_dir_if_not_exist(target_dir_path)
        for image_path in image_path_list:
            refer_image_path = image_path.replace(src_image_root_dir, refer_image_root_dir, 1)
            h, w = get_image_size(refer_image_path)
            resized_image_path = image_path.replace(src_image_root_dir, target_image_root_dir, 1)

            tmp_img = cv2.imread(image_path)
            resized_img = cv2.resize(tmp_img, (w, h))
            cv2.imwrite(resized_image_path, resized_img)
            # print(f'Generating empty file {txt_file_name} ...')
            count += 1
    print(f'Total {count} images resized ...')


def report_image_size(image_dir):
    """
    # report the statistics for the image information, e.g.,
    image_dir = 'data/driver_face/test_data/sample91_images'
    :param image_dir:
    :return:
    """
    image_list_dir = image_dir + '_image_list'
    generate_dir_if_not_exist(image_list_dir)
    dir_list = get_dir_list_in_directory(image_dir)

    video_image_size = []
    image_dir_for_detection = []
    for idx, dir_name in enumerate(dir_list):
        # print(f'Processing {dir_name} [{idx + 1}/{len(dir_list)}] ... ')
        # image_file_list = get_file_list_recursively(
        #     os.path.join(image_dir, dir_name)
        # )
        image_file_list = get_image_file_list_in_directory(
            os.path.join(image_dir, dir_name)
        )
        if len(image_file_list) == 0:
            print(f'{dir_name} has no images...')
            continue

        image_name = image_file_list[0]
        img = cv2.imread(image_name)
        if img is not None:
            height, width, _ = img.shape
            video_infor = ','.join(map(str, [dir_name, len(image_file_list), width, height]))
        else:
            video_infor = ','.join(map(str, [dir_name, len(image_file_list), 0, 0]))

        print(f'{video_infor}')
        video_image_size.append(video_infor)

        image_list_abs_path = [os.path.abspath(x) for x in image_file_list]
        file_name = os.path.join(image_list_dir, dir_name + '.txt')
        save_1d_list_to_file(file_name, image_list_abs_path)

        image_dir_for_detection.append(os.path.abspath(file_name))

    save_1d_list_to_file(image_dir + '_size_infor.txt', video_image_size)
    save_1d_list_to_file(os.path.join(image_list_dir, 'video_infor.txt'), image_dir_for_detection)
    # smrc.line.save_1d_list_to_file(
    #   os.path.join('sample91_images', 'video_infor.txt'),image_dir_for_detection
    # )


def count_num_lines(label_root_dir, ext_str='.txt'):
    """
    Count the number of lines in txt files.
    :param ext_str:
    :param label_root_dir:
    :return:
    """
    dir_list = get_dir_list_in_directory(label_root_dir)
    file_count = 0
    line_count = 0
    # dir_list = dir_list[0]
    pbar = tqdm(enumerate(dir_list))
    for dir_idx, dir_name in pbar:
        pbar.set_description(f'Processing {dir_name} ({dir_idx}/{len(dir_list)}) ...')
        # print(f'Processing {dir_name} ({dir_idx}/{len(dir_list)}) ...')
        txt_file_list = get_file_list_in_directory(
            os.path.join(label_root_dir, dir_name), ext_str=ext_str
        )
        for txt_file in txt_file_list:
            statinfo = os.stat(txt_file)
            # print(f'{txt_file}, {statinfo.st_size}')
            if statinfo.st_size > 0:
                ann_list = load_multi_column_list_from_file(txt_file)
                # print(f'{dir_name} {len(ann_list)}  ... ')
                file_count += 1
                line_count += len(ann_list)
    print(f'Total {file_count} non empty file, {line_count} lines ...')


#############################################
#
############################################


def move_and_rename_files_recursively(dir_path):
    """
    Move and rename the files in dir_path with depth > 1 to
    files with depth 1.
    For instance, 'test_data/0000.jpg', 'test_data/1/0000.jpg'
    will become
        'test_data/0000.jpg', 'test_data/1_0000.jpg'
    respectively
    :param dir_path:
    :return:
    """
    sub_dir_list = get_dir_list_in_directory(dir_path)
    # if there is no sub_dir_list
    if len(sub_dir_list) == 0:
        return
    else:
        file_list = []
        for sub_dir_name in sub_dir_list:
            file_list += get_file_list_recursively(
                    os.path.join(dir_path, sub_dir_name)
                )

        print(f'Total {len(file_list)} files need to be renamed ...')
        pbar = tqdm(file_list)
        for file in pbar:
            pbar.set_description(f'Processing {file} ...')
            str_id = file.find(dir_path) + len(dir_path) + 1
            new_basename = replace_dir_sep_with_underscore(
                file[str_id:]
            )
            os.rename(file, os.path.join(dir_path, new_basename))

        for sub_dir_name in sub_dir_list:
            shutil.rmtree(os.path.join(dir_path, sub_dir_name))


def multi_level_dir_to_two_level(data_root_dir):
    """
    Transfer a multi-level dir a two levels dir.
    This will be useful for transferring images or labels with multiple
    depth to 2-level depth for easy use.
    :param data_root_dir:
    :return:
    """

    dir_list = get_dir_list_in_directory(data_root_dir)
    for k, dir_name in enumerate(dir_list):
        print(f'Processing {k}/{len(dir_list)} dir {dir_name}')

        # move all the files in the dir_name to one level dir
        target_dir = os.path.join(data_root_dir, dir_name)
        move_and_rename_files_recursively(target_dir)

        # sub_dir_list = get_dir_list_in_directory(
        #     target_dir, only_local_name=False
        # )
        # for sub_dir in sub_dir_list:
        #     move_and_rename_files_recursively(sub_dir)


# def report_image_size_complete(image_dir):
#     """
#     # report the statistics for the image information, e.g.,
#     image_dir = 'data/driver_face/test_data/sample91_images'
#     :param image_dir:
#     :return:
#     """
#     image_list_dir = image_dir + '_image_list'
#     generate_dir_if_not_exist(image_list_dir)
#     dir_list = get_dir_list_in_directory(image_dir)
#
#     video_image_size = []
#     image_dir_for_detection = []
#     for idx, dir_name in enumerate(dir_list):
#         # print(f'Processing {dir_name} [{idx + 1}/{len(dir_list)}] ... ')
#         image_file_list = get_file_list_recursively(
#             os.path.join(image_dir, dir_name)
#         )
#         if len(image_file_list) == 0:
#             print(f'{dir_name} has no images...')
#             continue
#
#         image_name = image_file_list[0]
#         img = cv2.imread(image_name)
#         if img is not None:
#             height, width, _ = img.shape
#             video_infor = ','.join(map(str, [dir_name, len(image_file_list), width, height]))
#         else:
#             video_infor = ','.join(map(str, [dir_name,len(image_file_list), 0, 0]))
#
#         print(f'{video_infor}')
#         video_image_size.append(video_infor)
#
#         image_list_abs_path = [os.path.abspath(x) for x in image_file_list]
#         file_name = os.path.join(image_list_dir, dir_name + '.txt')
#         save_1d_list_to_file(file_name, image_list_abs_path)
#
#         image_dir_for_detection.append(os.path.abspath(file_name))
#
#     save_1d_list_to_file(image_dir + '_size_infor.txt', video_image_size)
#     save_1d_list_to_file(os.path.join(image_list_dir, 'video_infor.txt'), image_dir_for_detection)
#     # smrc.line.save_1d_list_to_file(
#     #   os.path.join('sample91_images', 'video_infor.txt'),image_dir_for_detection
#     # )1