import os
import cv2
# import random
# import numpy as np
from tqdm import tqdm
import glob
import imagesize
import filetype
import shutil

from .color import *
from .base import generate_dir_if_not_exist, get_file_list_in_directory, \
    get_file_list_recursively, get_dir_list_in_directory, natural_sort_key


def test_filetype():
    """
    <filetype.types.image.Jpeg object at 0x7f451a4b60d0>
    File extension: jpg
    File MIME type: image/jpeg
    <filetype.types.video.Avi object at 0x7f451570b9d0>
    File extension: avi
    File MIME type: video/x-msvideo
    :return:
    """
    filename = "tests.jpg"
    kind = filetype.guess(filename)
    print(kind)
    print('File extension: %s' % kind.extension)
    print('File MIME type: %s' % kind.mime)
    # print(filetype.guess(filename) in filetype.types.image)
    # <filetype.types.image.Jpeg object at 0x7fa7e6542190>
    filename = "output.avi"  # <filetype.types.video.Avi object at 0x7fae65be9990>
    kind = filetype.guess(filename)
    print(kind)
    print('File extension: %s' % kind.extension)
    print('File MIME type: %s' % kind.mime)


def is_image(filename):
    kind = filetype.guess(filename)
    if kind is not None:
        file_type_str = kind.mime
        return file_type_str.find('image') > -1
    else:
        return False


def is_video(filename):
    kind = filetype.guess(filename)
    if kind is not None:
        file_type_str = kind.mime
        return file_type_str.find('video') > -1
    else:
        return False


def generate_blank_image(height, width, color=WHITE):
    img = np.zeros((height, width, 3), np.uint8)   # white background
    img[:, :] = color

    return img


def get_image_size(image_path):
    """
    Return height, width for a given image path.
    :param image_path:
    :return:
    """
    assert os.path.isfile(image_path)
    # tmp_img = cv2.imread(image_path)
    # assert tmp_img is not None
    # height, width, _ = tmp_img.shape

    width, height = imagesize.get(image_path)
    return height, width


def get_image_file_list_in_directory(directory_path):
    """Get the image file by is_image (function from filetype) and thus it is a fast version
    than cv2.imread()
    The directory_path + image_file_name will be returned.
    :param directory_path:
    :return:
    """
    file_path_list = []
    # load image list
    for f in sorted(os.listdir(directory_path), key=natural_sort_key):
        f_path = os.path.join(directory_path, f)
        if os.path.isdir(f_path):
            # skip directories
            continue
        # check if it is an image
        if is_image(f_path):
            file_path_list.append(f_path)
    return file_path_list


def rename_image_file_list_in_directory(
        directory_path, directory_path_new=None, num_digits=4
):
    if directory_path_new is not None:
        generate_dir_if_not_exist(directory_path_new)

    format_str = '0' + str(num_digits) + 'd'  # e.g., '04d'

    file_path_list = []
    # load image list
    image_id = 0
    for f in sorted(os.listdir(directory_path), key=natural_sort_key):
        f_path = os.path.join(directory_path, f)
        if os.path.isdir(f_path):
            # skip directories
            continue

        # check if it is an image
        # test_img = cv2.imread(f_path)
        # if test_img is not None:
        if is_image(f_path):
            file_path_list.append(f_path)
            if directory_path_new is None:
                new_image_name = os.path.join(directory_path, format(image_id, format_str) + '.jpg')
            else:
                new_image_name = os.path.join(directory_path_new, format(image_id, format_str) + '.jpg')
            image_id += 1
            # cv2.imwrite(new_image_name, test_img)
            shutil.copyfile(f_path, new_image_name)
    return file_path_list


def image_size_for_image_sequence(image_sequence_dir):
    """Get the image size for a video (a list of image_sequence)

    :param image_sequence_dir:
    :return:
    """
    assert os.path.isdir(image_sequence_dir)
    image_path_list = get_file_list_recursively(image_sequence_dir)
    assert len(image_path_list) > 0
    # image_path = image_path_list[0]
    # tmp_img = cv2.imread(image_path)
    # height, width = tmp_img.shape[:2]
    # return height, width

    return get_image_size(image_path_list[0])


def convert_frames_to_video_with_cut_background(background_image_filename, pathIn, pathOut, fps):
    background_image = cv2.imread(background_image_filename)
    print(background_image.shape)
    background_height, background_width, _ = background_image.shape
    frame_array = []
    # 'visualization/image/2595/0000.jpg'
    files = get_file_list_in_directory(pathIn)

    size = None
    for filename in files:
        # print('filename = ' + filename)
        # reading each files
        img = cv2.imread(filename)
        if img is not None:
            height, width, layers = img.shape
            size = (width, height)
            # print(filename)
            # tmp_img[ymin:ymax, xmin:xmax]
            img[50:50 + background_height, 550:550 + background_width] = background_image
            # inserting the frames into an image array
            frame_array.append(img)
            image_name_new = filename.replace(pathIn, pathOut, 1)
            cv2.imwrite(image_name_new, img)

    if len(frame_array) > 1:
        out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

        for i in range(len(frame_array)):
            # writing to a image array
            out.write(frame_array[i])
        out.release()
    else:
        print('There are only %d image frames in %s, failed to generate video.' % (len(frame_array), pathIn))


def convert_frames_to_video(pathIn, pathOut, fps):
    # frame_array = []
    # 'visualization/image/2595/0000.jpg'
    files = get_file_list_in_directory(pathIn)

    if len(files) == 0:
        return
    # assert len(files) > 0
    height, width = get_image_size(files[0])
    size = (width, height)
    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    print(f'Generating {os.path.abspath(pathOut)} '
          f'from {len(files)} images in {os.path.abspath(pathIn)} ...')
    for i, filename in enumerate(tqdm(files)):
        # reading each files
        img = cv2.imread(filename)
        if img is not None:
            # height, width, layers = img.shape
            # size = (width,height)
            # print(filename)
            out.write(img)
            # inserting the frames into an image array
            # frame_array.append(img)
    out.release()


def convert_frames_to_video_for_given_image_list(image_list, pathOut, fps, frames_per_video=None):
    max_height, max_width = 0, 0
    for filename in image_list:
        # print('filename = ' + filename)
        # reading each files

        height, width = get_image_size(filename)
        if height > max_height:
            max_height = height

        if width > max_width:
            max_width = width
    size = (max_width, max_height)

    background_img = np.zeros((max_height, max_width, 3), np.uint8)  # white background
    background_img[:, :] = [255, 255, 255]
    x0, y0 = float(max_width / 2.0), float(max_height / 2.0)

    # if there is too much images, we try to control the size of the images for video generating
    # otherwise, the program will die to the memory issue.
    if frames_per_video is None:
        frames_per_video = len(image_list)  # generate only one video
    elif frames_per_video < 2:  # if the given frames_per_video is not valid
        frames_per_video = 120

    # pathOut = 'tests.jpg'
    # prefix, ext = 'tests', 'jpg'
    # to avoid the issues caused by relative path '.', '..'
    prefix, ext = os.path.abspath(pathOut).split('.')
    num_videos = int(len(image_list) / (frames_per_video * fps)) + 1
    print('num_videos = %d' % (num_videos,))
    for video_id in range(num_videos):  # frames_per_video sec, iamge frame length for this video
        frame_array = []
        video_name = prefix + str(video_id) + '.' + ext
        if video_id < num_videos - 1:
            files_to_process = image_list[frames_per_video * fps * video_id: frames_per_video * fps * (video_id + 1)]
        else:
            files_to_process = image_list[frames_per_video * fps * video_id: -1]

        print('Precoesing video %s ' % video_name)
        for filename in files_to_process:
            # print('filename = ' + filename)
            # reading each files
            img = cv2.imread(filename)
            if img is not None:
                height, width, layers = img.shape
                frame_image = background_img.copy()
                ymin, ymax = int(y0 - height / 2.0), int(y0 + height / 2.0)
                xmin, xmax = int(x0 - width / 2.0), int(x0 + width / 2.0)
                frame_image[ymin:ymax, xmin:xmax] = img

                # if generate the new padding image frames, then uncoment and modify the following sentences.
                # filename_new = filename.replace('bbox_sorted', 'bbox_sorted_same_size', 1)
                # cv2.imwrite(filename_new, frame_image)

                ##inserting the frames into an image array
                frame_array.append(frame_image)

        if len(frame_array) > 1:
            out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

            for i in range(len(frame_array)):
                # writing to a image array
                out.write(frame_array[i])
            out.release()
        else:
            print('There are only %d image frames, failed to generate video.' % (len(frame_array)))


def convert_frames_to_video_different_size(pathIn, pathOut, fps, frames_per_video=None):
    '''
    generate videos for images of different size.
    if the there are too many images, we will generate multiple videos, and one videos
    contains only 'frames_per_video' images

    '''

    # here we assure the files under pathIn are only images, e.g.,
    #  'visualization/image/2595/0000.jpg'
    # if this is not the case, then we need to make a new function
    files = get_file_list_in_directory(pathIn)
    convert_frames_to_video_for_given_image_list(files, pathOut, fps, frames_per_video)


def convert_frames_to_video_for_all_images_in_parent_dir(pathIn, pathOut, fps, frames_per_video, shuffle_option=True):
    image_list = get_file_list_recursively(pathIn)

    if shuffle_option:
        random.shuffle(image_list)
    # print(pathIn)
    # print(image_list[0:100])
    # sys.exit(0)
    # image_list = image_list[0:200]

    convert_frames_to_video_for_given_image_list(image_list, pathOut, fps, frames_per_video)


def convert_frames_to_video_inside_directory(pathInParentDir, fps, ext_str='.avi', with_str=None):
    '''
    convert the images for each sub dir of the directories under 'parentDir' to videos

    examples,
        self.convert_frames_to_video_inside_directory('truck_images', 30)

    will generate videos under 'truck_images' for each of its sub directory
    '''
    if with_str is None:
        dir_list = get_dir_list_in_directory(pathInParentDir, only_local_name=False)
    else:
        dir_list = glob.glob(os.path.join(pathInParentDir, '*' + with_str + '*'))
    for i, pathIn in enumerate(dir_list):
        print(f'{i+1}/{len(dir_list)} video ...')
        pathOut = pathIn + ext_str  # truck_images/1 -> truck_images/1.avi
        convert_frames_to_video(pathIn, pathOut, fps)


# code from https://www.life2coding.com/convert-image-frames-video-file-using-opencv-python/
def convert_video_to_frames(pathIn, pathOut, prefix=None, fps=None):
    '''Some of the frames can be skipped if fps > 30.
    E.g., if fps = 60, we will skip one frame for every two frames.
    convert a video to image frames
        pathIn, video name, e.g., video.avi
        pathOut, directory name of the resulting image frames
    example,
        convert_video_to_frames('dataset/video.avi', 'video')
    '''
    generate_dir_if_not_exist(pathOut)
    # Opens the Video file
    cap = cv2.VideoCapture(pathIn)
    print(f'Converting {pathIn} to {pathOut} ...')
    if fps is not None and fps > 30:
        times = int(fps / 30)
        # print(f'times = {times}')
    else:
        times = 1
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if prefix is not None:
            image_name = os.path.join(pathOut, prefix + format(i, '06d') + '.jpg')
        else:
            image_name = os.path.join(pathOut, format(i, '06d') + '.jpg')
        # skip some frames if we wish
        if i % times == 0:
            # print(f'i = {i}, fps = {fps}, {image_name}')
            cv2.imwrite(image_name, frame)
        i += 1
    cap.release()


def convert_multiple_videos_to_frames(dir_path):
    video_list = get_file_list_in_directory(dir_path)
    # print(video_list)
    for video_name in tqdm(video_list):
        pathOut = video_name[:video_name.rfind('.')]
        # print(pathOut)
        convert_video_to_frames(video_name, pathOut)
