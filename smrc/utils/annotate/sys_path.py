import os
import argparse
from .. import base

# smrc/line/annotate/.
file_path = os.path.abspath(__file__)


SMRC_ROOT_PATH = base.dir_name_up_n_levels(
    file_path, 3
)
# # /home/kai/tuat/smrc
# print(SMRC_ROOT_PATH)


def argparse_visualization():
    parser = argparse.ArgumentParser(description='SMRC')
    parser.add_argument('-i', '--image_dir', default='images', type=str, help='Path to image directory')
    parser.add_argument('-l', '--label_dir', default='labels', type=str, help='Path to label directory')
    parser.add_argument('-c', '--class_list_file', default='class_list.txt', type=str,
                        help='File that defines the class labels')
    parser.add_argument('-u', '--user_name', default=None, type=str, help='User name')
    # either 'label_dir' or 'image_dir'
    parser.add_argument('-a', '--auto_load_directory', default=None, type=str,
                        help='The root dir for directories to load')
    args = parser.parse_args()

    AUTO_LOAD_DIRECTORY = None
    if args.auto_load_directory is not None:
        if args.auto_load_directory.find('image') >= 0:
            AUTO_LOAD_DIRECTORY = 'image_dir'
        elif args.auto_load_directory.find('label') >= 0:
            AUTO_LOAD_DIRECTORY = 'label_dir'
        else:
            print(f'AUTO_LOAD_DIRECTORY should be in "image_dir", "label_dir"')

    user_name = args.user_name
    image_dir = args.image_dir
    label_dir = args.label_dir
    class_list_file = args.class_list_file
    auto_load_directory = AUTO_LOAD_DIRECTORY
    return image_dir, label_dir, class_list_file, user_name, auto_load_directory





