# basic line used all through various tools
import re
import sys
import os
import csv
import random
import collections
import pickle

# import cv2
# import numpy as np
# from .dir_file import get_file_list_recursively


def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(s)]


def non_blank_lines(f):
    for l in f:
        line = l.rstrip()
        if line:
            yield line


def flattened_2d_list(my_list):
    return [x for sublist in my_list for x in sublist]


def unique_element_in_2d_list(my_list):
    return [list(t) for t in set(tuple(element) for element in my_list)]


def time_stamp_str():
    """
    >> print(time.strftime("%Y-%m-%d-%H-%M-%S"))
    2019-11-18-20-49-22
    :return:
    """
    import time
    return time.strftime("%Y-%m-%d-%H-%M-%S")


######################################################
# load/save list from/to file
######################################################

def load_1d_list_from_file(filename):
    assert os.path.isfile(filename)

    with open(filename) as f_directory_list:
        resulting_list = list(non_blank_lines(f_directory_list))

    return resulting_list


def load_class_list_from_file(class_list_file):
    # class_list_file = os.path.join(
    #     os.path.dirname(os.path.abspath(__file__)), class_list_file
    # )
    with open(class_list_file) as f:
        class_list = list(non_blank_lines(f))
    return class_list


def load_directory_list_from_file(filename):
    return load_1d_list_from_file(filename)


def load_multi_column_list_from_file(filename, delimiter=','):

    resulting_list = []
    assert os.path.isfile(filename)

    with open(filename, 'r') as old_file:
        lines = list(non_blank_lines(old_file))

    # print('lines = ',lines)
    for line in lines:
        line = line.strip()
        result = line.split(delimiter)

        resulting_list.append(result)
    return resulting_list


def save_1d_list_to_file(file_path, list_to_save):
    with open(file_path, 'w') as f:
        for item in list_to_save:
            f.write("%s\n" % item)


def save_multi_dimension_list_to_file(filename, list_to_save, delimiter=','):
    with open(filename, 'w') as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerows(list_to_save)  # considering my_list is a list of lists.


######################################################
# get file list or dir list
######################################################

def assert_dir_exist(dir_name):
    if not os.path.isdir(dir_name):
        print(f'Directory {os.path.abspath(dir_name)} not exist, please check ...')
        sys.exit(0)


def get_dir_list_recursively(walk_dir):
    """
    for root, subdirs, files in os.walk(rootdir):
    root: Current path which is "walked through"
    subdirs: Files in root of type directory
    files: Files in root (not in subdirs) of type other than directory

    dir_list.append( subdir ) does not make sense
    e.g., tests/test1 tests/test1/test2
    if dir_list.append( subdir ) will return tests, test1, test2

    , only_local_name=False
    """
    assert os.path.isdir(walk_dir)

    dir_list = []
    # https://www.mkyong.com/python/python-how-to-list-all-files-in-a-directory/
    for root, subdirs, files in os.walk(walk_dir):
        for subdir in subdirs:
            # if only_local_name:
            #     dir_list.append(subdir)
            # else:
            dir_list.append(os.path.join(root, subdir))
    return dir_list


def get_file_list_recursively(root_dir, ext_str=''):
    """
    ext_str = None (not specified, then files)
    ext_str: suffix for file, '.jpg', '.txt'  , only_local_name=False
    """
    assert os.path.isdir(root_dir)

    file_list = []  # the full relative path from root_dir

    for root, subdirs, files in os.walk(root_dir):
        # print(subdirs)
        # print(files) # all files without dir name are saved

        for filename in files:
            if ext_str in filename:
                # if only_local_name:
                #     file_list.append(filename)
                # else:
                file_list.append(os.path.join(root, filename))
    # if not sort, then the images are not ordered.
    # 'visualization/image/3473/0257.jpg',
    #  'visualization/image/3473/0198.jpg',
    # 'visualization/image/3473/0182.jpg',
    # 'visualization/image/3473/0204.jpg'
    # file_list.sort()
    file_list = sorted(file_list, key=natural_sort_key)
    return file_list


def get_relative_file_list_recursively(root_dir, ext_str=''):
    file_list = get_file_list_recursively(root_dir, ext_str=ext_str)
    if len(file_list) > 0:
        for k, file in enumerate(file_list):
            file_list[k] = extract_relative_file_path(file, root_dir)
    return file_list


def extract_relative_file_path(file_path, root_dir):
    str_id = file_path.find(root_dir) + len(root_dir) + 1
    return file_path[str_id:]


def get_dir_list_in_directory(directory_path, only_local_name=True):
    """
    list all the directories under given 'directory_path'
    return a list of full path dir, in terms of
            directory_path  + sub_dir_name

    e.g.,
        get_dir_list_in_directory('truck_images')
        return
            ['truck_images/1', 'truck_images/2', ... ]
    """
    assert os.path.isdir(directory_path), f'Do not exist [{os.path.abspath(directory_path)}] ...'

    dir_path_list = []
    for f in sorted(os.listdir(directory_path), key=natural_sort_key):
        f_path = os.path.join(directory_path, f)  # images/2
        if os.path.isdir(f_path):
            if only_local_name:
                dir_path_list.append(f)
            else:
                dir_path_list.append(f_path)
    return dir_path_list


def get_file_list_in_directory(
        directory_path, only_local_name=False, ext_str=''
):
    # print(f'only_local_name = {only_local_name}, ext_str = {ext_str} ...')
    assert os.path.isdir(directory_path)

    file_path_list = []
    # load image list
    for f in sorted(os.listdir(directory_path), key=natural_sort_key):
        f_path = os.path.join(directory_path, f)
        if os.path.isdir(f_path) or f.find(ext_str) == -1:
            # skip directories
            continue
        else:
            if only_local_name:
                file_path_list.append(f)
            else:
                file_path_list.append(f_path)
    return file_path_list
    # return directory_path


def get_json_file_list(directory_path, only_local_name=False):
    return get_file_list_in_directory(
        directory_path, only_local_name=only_local_name, ext_str='.json')


def generate_dir_if_not_exist(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def generate_dir_list_if_not_exist(dir_list):
    for dir_name in dir_list:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)


def replace_same_level_dir(reference_dir_path, target_dir_path):
    """
    replace the last level dir path of "reference_dir_path" with "target_dir_path",
     and generate the "target_dir_path" in the same level with "reference_dir_path"
    :param reference_dir_path: e.g., images
    :param target_dir_path: e.g., labels
    :return:
    """
    return os.path.join(os.path.dirname(reference_dir_path), target_dir_path)
    # generate_dir_if_not_exist(resulting__dir)



######################################################
# file path operation
######################################################


def file_path_last_two_level(file_path):
    file_names = file_path.split(os.path.sep)
    assert len(file_names) >= 2
    return os.path.join(file_names[-2], file_names[-1])


def extract_last_file_name(file_path):
    return file_path.split(os.path.sep)[-1]


def get_image_or_annotation_path(oldFilename, oldDir, newDir, newExt):
    old_path = oldFilename.replace(oldDir, newDir, 1)
    _, oldExt = os.path.splitext(old_path)
    new_path = old_path.replace(oldExt, newExt, 1)
    return new_path


def append_suffix_to_file_path(old_path, suffix):
    _, oldExt = os.path.splitext(old_path)
    newExt = suffix + oldExt
    new_path = old_path.replace(oldExt, newExt, 1)
    return new_path


def append_prefix_to_file_path(old_path, prefix):
    dir_name = os.path.dirname(old_path)
    basename = os.path.basename(old_path)
    new_path = os.path.join(
        dir_name, prefix + basename
    )
    return new_path


def replace_ext_str(old_path, new_ext_str):
    filename, oldExt = os.path.splitext(old_path)
    new_path = filename + new_ext_str
    return new_path

###########################################
# other often used operations in our project
############################################


def repeated_element_in_list(list_to_to_check):
    return [item for item, count in collections.Counter(list_to_to_check).items()
            if count > 1]


def repeated_element_and_count_in_list(list_to_to_check):
    return [(item, count) for item, count in collections.Counter(list_to_to_check).items()
            if count > 1]


def shared_list(list1, list2):
    return repeated_element_in_list(list1 + list2)


def diff_list(list1, list2):
    unique_list1 = [x for x in list1 if x not in list2]
    unique_list2 = [x for x in list2 if x not in list1]
    print(f'len(set(list1)) = {len(set(list1))}, len(set(list2)) = {len(set(list2))}')
    print(f'Repeated element in list1 {repeated_element_and_count_in_list(list1)} ... ')
    print(f'Repeated element in list2 {repeated_element_and_count_in_list(list2)} ... ')
    print("Difference: Only in list1 \n", set(list1) - set(list2))
    print("Difference: Only in list2 \n", set(list2) - set(list1))

    return unique_list1, unique_list2


def train_test_split_video_list(video_list, num_test, random_state=None):
    if random_state is not None:
        random.seed(random_state)

    random.shuffle(video_list)
    test_video_list, train_video_list = video_list[:num_test], video_list[num_test:]
    return train_video_list, test_video_list


def generate_train_test_video_list(label_dir, num_test, random_state=None):
    dir_list = get_dir_list_in_directory(label_dir)

    train_video_list, test_video_list = train_test_split_video_list(dir_list, num_test, random_state)

    train_list_filename = os.path.join(
        os.path.dirname(label_dir),
        f'train_{len(train_video_list)}video_list'
    )
    test_list_filename = os.path.join(
        os.path.dirname(label_dir),
        f'test_{len(test_video_list)}video_list'
    )

    save_1d_list_to_file(train_list_filename, train_video_list)
    save_1d_list_to_file(test_list_filename, test_video_list)
    return train_video_list, test_video_list


def exclude_one_list_from_another(full_dir_list, dir_list_to_exclude):
    remaining_video_list = []
    for video in full_dir_list:
        if video not in dir_list_to_exclude:
            remaining_video_list.append(video)

    return remaining_video_list


def exclude_one_file_list_from_another_file(
        full_dir_list_file, dir_list_to_exclude_file,
        resulting_file=None
):

    video_list1 = load_1d_list_from_file(full_dir_list_file)

    video_list2 = load_1d_list_from_file(dir_list_to_exclude_file)

    remaining_video_list = exclude_one_list_from_another(video_list1, video_list2)

    if resulting_file is not None:
        save_1d_list_to_file(resulting_file, remaining_video_list)

    return remaining_video_list


##################################
#
#######################################


def load_pkl_file(pkl_file_name):
    if os.path.isfile(pkl_file_name):
        f = open(pkl_file_name, 'rb')
        data = pickle.load(f)
        f.close()
        print(f'data loaded from {pkl_file_name} .')
        return data
    else:
        return None


def generate_pkl_file(pkl_file_name, data, show_done=True):
    f = open(pkl_file_name, 'wb')
    pickle.dump(data, f)
    f.close()
    if show_done:
        print(f'{pkl_file_name} saved ...')

    # with open(pkl_file_name, 'wb') as f:
    #     pickle.dump(data, f)

#################################
# path operation
######################################


def dir_name_up_n_levels(file_abspath, n):
    k = 0
    while k < n:
        file_abspath = os.path.dirname(file_abspath)
        k += 1

    return file_abspath


def specify_dir_list(data_root_dir, dir_list=None):
    if dir_list is None:
        dir_list = get_dir_list_in_directory(data_root_dir, only_local_name=True)
    return dir_list
