import cv2
import sys
import os
import numpy as np
from .base import generate_dir_if_not_exist


def merging_two_images_side_by_side(img1, img2):
    """
    convert two images to one side by side
    modified from
    https://stackoverflow.com/questions/7589012/combining-two-images-with-opencv

    :param img1:
    :param img2:
    :return:
    """
    if img1 is None or img2 is None:
        print('img1 or img2 is None, please check...')
        sys.exit(0)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Create an array big enough to hold both images next to each other.
    img = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)

    # Copy both images into the composite image.
    img[:h1, :w1, :] = img1
    img[:h2, w1:w1 + w2, :] = img2

    # cv2.imshow('tests', img)
    return img


def test_merging_images():
    img1 = cv2.imread('round0_object_0_0000.jpg')
    img2 = cv2.imread('round0_object_0_0001.jpg')
    img_merged = merging_two_images_side_by_side(img1, img2)
    cv2.imwrite('tests.jpg', img_merged)


def plot_image_difference(image_list, result_dir=None):
    for i in range(len(image_list)-1):
        image_name1 = image_list[i]
        image_name2 = image_list[i+1]

        img1, img2 = cv2.imread(image_name1), cv2.imread(image_name2)
        diff = img1 - img2

        # print(diff[0,0,:10)
        print(f'Saving difference image for {image_name1} and {image_name2}')
        if result_dir is not None:
            generate_dir_if_not_exist(result_dir)
            cv2.imwrite(
                os.path.join(result_dir, 'diff' + str(i) + '.jpg', diff)
            )
        else:
            cv2.imwrite('diff' + str(i) + '.jpg', diff)


def show_img(frame):
    cv2.imshow('Showing image for debug', frame)

    while True:
        # Press Q to stop!
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break
