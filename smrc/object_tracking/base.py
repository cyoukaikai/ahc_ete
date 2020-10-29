import os
import sys
import cv2
import random
import numpy as np


def replace_dir_sep_with_underscore(dir_name):
    return dir_name.replace(os.path.sep, '_')


def save_matrix_to_txt(filename, mat, num_digit):
    # np.savetxt('final_similar_matrix', final_similar_matrix, fmt='%.3f')
    np.savetxt(filename, mat, fmt=f'%.{num_digit}f')

