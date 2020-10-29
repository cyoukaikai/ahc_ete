import os
import cv2
import re
import json
import numpy as np
import sys
import random
import shutil
import smrc.utils
from .annotation_tool import AnnotationTool


#########################################################
# visualize the object_detection or annotation result in
# terms of images with bounding boxes, or videos with detections or annotations
#########################################################


class ExportDetection(AnnotationTool):
    def __init__(self, image_dir,
                 label_dir,
                 class_list_file,
                 visualization_option=None,  # ['image']
                 visualization_result_dir=None,
                 skip_non_empty_image=False,
                 with_blank_bg=False,
                 # continue_last_visualization=True,
                 fps=30,
                 directory_list_to_visualize=None  # None means we do not specify which dir to visualize, i.e., 'all'
                 ):
        assert class_list_file is not None and os.path.isfile(class_list_file)

        AnnotationTool.__init__(self, image_dir=image_dir, label_dir=label_dir, class_list_file=class_list_file)
        self.IMAGE_DIR = image_dir
        self.LABEL_DIR = label_dir

        self.class_index_list = list(range(len(self.CLASS_LIST)))  # the class indices we used
        self.LINE_THICKNESS = 2
        self.image_to_video_fps = fps  # default value 30
        self.skip_non_empty_image = skip_non_empty_image
        self.with_blank_bg = with_blank_bg

        if visualization_option is None:
            visualization_option = ['image']

        if visualization_result_dir is None:
            visualization_result_dir = smrc.utils.replace_same_level_dir(
                reference_dir_path=label_dir,
                target_dir_path='visualization'
            )

        print('====================================================')
        print(f'To generate images or videos for {self.LABEL_DIR}.')
        print(f'The images in {self.IMAGE_DIR} will be used to assist the visualization.')
        print(f'self.with_blank_bg = {self.with_blank_bg} ...')
        print(f'The final results will be saved in {visualization_result_dir}.')
        print('====================================================')

        self.visualization_annotation_result(
            self.LABEL_DIR,
            visualization_result_dir,
            visualization_option,
            directory_list_to_visualize
        )
        ##self.visualize_annotated_bbox_large_to_small(visualization_result_dir)

    def visualize_annotated_bbox_large_to_small(self, visualization_result_dir):
        parent_dir_name = os.path.join(visualization_result_dir, 'bbox_sorted')
        smrc.utils.generate_dir_if_not_exist(parent_dir_name)
        smrc.utils.generate_dir_if_not_exist(os.path.join(visualization_result_dir, 'bbox_sorted_same_size'))

        print('To visualize the annotated bbox of {} classes.'.format(len(self.class_index_list)))
        for class_idx, class_name in enumerate(self.CLASS_LIST):
            print('Processing class: ' + class_name)
            dir_class_name_sorted = os.path.join(*[parent_dir_name,
                                                   class_name])
            smrc.utils.generate_dir_if_not_exist(dir_class_name_sorted)

            dir_class_name = os.path.join(*[visualization_result_dir, 'bbox',
                                            class_name])
            # the bbox sorted but with same size by padding white background.
            smrc.utils.generate_dir_if_not_exist(
                os.path.join(*[visualization_result_dir, 'bbox_sorted_same_size',
                               class_name])
            )

            image_file_list = smrc.utils.get_file_list_in_directory(dir_class_name)
            # image_file_list = image_file_list[0:1000]
            print('Estimating image areas for class %s, total %d subimages: ' % (class_name, len(image_file_list)))
            area_list = []
            bbox_image_name_old = []
            for file_idx, image_path in enumerate(image_file_list):

                # generate new image name based on image size
                print('processing %dth image, name %s' % (file_idx, image_path))
                tmp_img = cv2.imread(image_path)
                if tmp_img is not None:
                    height, width, layers = tmp_img.shape
                    area = width * height
                    area_list.append(area)
                    bbox_image_name_old.append(image_path)
                else:
                    print(' Can not load image {}'.format(image_path))

            # print('area_list before sort =', area_list)
            # sort the area list
            sorted_area_list_index = sorted(range(len(area_list)),
                                            key=lambda k: area_list[k],
                                            reverse=True)  # sort the list based on its first element

            # print('sorted_area_list_index after sort =', sorted_area_list_index)
            print('Estimating image areas for class %s done. ' % (class_name,))

            # sorted_area_list_index = sorted_area_list_index[0:1000]
            # bbox_element_to_add = [area,bbox_id_str, bbox_image_name]
            for id, ele_id in enumerate(sorted_area_list_index):
                area, bbox_image_name = area_list[ele_id], bbox_image_name_old[ele_id]
                bbox_id_str_begin_index = bbox_image_name.find(class_name) + len(class_name) + 1
                bbox_id_str = bbox_image_name[bbox_id_str_begin_index:bbox_id_str_begin_index + 8]  # "%08d"
                new_bbox_id = "%08d" % (id,)
                bbox_image_name_new = bbox_image_name.replace(bbox_id_str, new_bbox_id, 1)
                bbox_image_name_new = bbox_image_name_new.replace(dir_class_name, dir_class_name_sorted, 1)

                shutil.copy(bbox_image_name, bbox_image_name_new)
                # print('class  = %d, bbox_area = %d, bbox_id = %s, new_bbox_id = %s, bbox_image_name =  %s bbox_image_name_new =  %s' %
                # (class_idx, area, bbox_id_str, new_bbox_id, bbox_image_name, bbox_image_name_new)
                # )

            pathIn = dir_class_name_sorted
            pathOut = os.path.join(*[parent_dir_name,
                                     class_name + '.avi'])
            smrc.utils.convert_frames_to_video_different_size(pathIn, pathOut, 5)  # self.image_to_video_fps

    def visualization_annotation_result(self, annotation_or_detection_dir, visualization_result_dir,
                                        visualization_option, directory_list_to_visualize=None):
        # visualization_option, datatype = list: element(s) in ['image', 'video', 'bbox']
        smrc.utils.generate_dir_if_not_exist(visualization_result_dir)  # generate 'visualization'

        if len(visualization_option) == 0:
            return

        for opt in visualization_option:
            smrc.utils.generate_dir_if_not_exist(os.path.join(visualization_result_dir, opt))

        if 'bbox' in visualization_option:
            bbox_dict_per_class = {}
            class_ids = []  # the class index that actually appears in the annotated txt files
            for class_idx in self.class_index_list:
                dir_class_name = os.path.join(*[visualization_result_dir, 'bbox',
                                                self.CLASS_LIST[class_idx]])
                smrc.utils.generate_dir_if_not_exist(dir_class_name)

        if directory_list_to_visualize is None:
            directory_list = smrc.utils.get_dir_list_in_directory(annotation_or_detection_dir)
        else:
            directory_list = directory_list_to_visualize

        #  parent_dir_name = 'visualization/image'
        parent_dir_name = os.path.join(visualization_result_dir, 'image')
        print('To visualize {} directories.'.format(len(directory_list)))
        for dir_idx, dir_name in enumerate(directory_list):
            # anno_dir_path = 'final_results_checked/25'
            image_dir_path = os.path.join(self.IMAGE_DIR, dir_name)
            anno_dir_path = os.path.join(annotation_or_detection_dir, dir_name)
            print('Precessing directory: {}, {}/{}'.format(anno_dir_path, str(dir_idx + 1),
                                                           str(len(directory_list))))
            # result_dir = 'visualization/image/25'
            result_dir = os.path.join(parent_dir_name, dir_name)
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            image_path_list = smrc.utils.get_file_list_in_directory(image_dir_path)
            # print(image_path_list)
            # load the txt file (annotation file) under the directory
            # txt_file_list = smrc.line.get_file_list_in_directory(anno_dir_path)
            for file_idx, image_path in enumerate(image_path_list):
                ann_path = self.get_annotation_path(image_path)
                # skip generating empty images
                if self.skip_non_empty_image:
                    if not os.path.isfile(ann_path) or len(list(smrc.utils.load_bbox_from_file(ann_path))) == 0:
                        # print(f'skip {ann_path}, not exist or empty.')
                        continue
                        # print(list(smrc.line.load_bbox_from_file(ann_path)))

                # image_path = 'images/3150/0447.jpg'
                # image_path = self.get_image_path_from_annotation_path(ann_path)
                # print('image_path =' + image_path)
                # check if it is an image
                tmp_img = cv2.imread(image_path)
                if tmp_img is not None:
                    # img_name = 0447.jpg
                    if 'bbox' in visualization_option:
                        img_name = image_path[image_path.find(dir_name) + len(dir_name) + 1:]
                        annotated_bboxes = smrc.utils.load_bbox_from_file(ann_path)
                        for bbox_idx, bbox in enumerate(annotated_bboxes):
                            class_idx, xmin, ymin, xmax, ymax = bbox
                            bbox_image_name_suffix = str(class_idx) + '_' + dir_name + '_' + img_name
                            bbox_image = tmp_img[ymin:ymax, xmin:xmax]

                            # generate the bbox image name
                            dir_class_name = os.path.join(*[visualization_result_dir, 'bbox',
                                                            self.CLASS_LIST[class_idx]])
                            if class_idx in bbox_dict_per_class.keys():
                                bbox_id_str = "%08d" % (bbox_dict_per_class[class_idx],)
                            else:
                                bbox_id_str = "%08d" % (0,)
                            bbox_image_name = os.path.join(dir_class_name,
                                                           bbox_id_str + '_' + bbox_image_name_suffix)

                            # print('class  = %d, bbox_area = %d, bbox_id_str = %s, bbox_image_name =  %s' %
                            # (class_idx, area, bbox_id_str, bbox_image_name)
                            # )
                            cv2.imwrite(bbox_image_name, bbox_image)

                            if class_idx not in class_ids:
                                class_ids.append(class_idx)
                                bbox_dict_per_class[class_idx] = 0
                            else:
                                bbox_dict_per_class[class_idx] += 1
                    if 'image' in visualization_option:
                        # plotting annotated bbox must be after extract the bbox regions, otherwise, the extracted
                        # bbox include the rectangle and class names
                        if self.with_blank_bg:  # replace the original image with blank background
                            height, width, _ = tmp_img.shape
                            tmp_img = smrc.utils.generate_blank_image(height, width)

                        tmp_img_with_bbox = tmp_img.copy()  # we have to copy tmp_img, otherwise, any modification
                        # of it will be saved in the extract subimage in visualization/bbox
                        self.draw_bboxes_from_file(tmp_img_with_bbox, ann_path)
                        # draw the annotated bbox on the image

                        # dir_old = 'images'
                        dir_old = image_path[0:image_path.find(dir_name) - 1]

                        # result_image_filename = 'visualization/image/3150/0448.jpg'
                        # 'images/3150/0447.jpg' -> 'visualization/image/3150/0448.jpg'
                        # replace 'images' -> 'visualization/image'
                        result_image_filename = image_path.replace(dir_old, parent_dir_name, 1)
                        cv2.imwrite(result_image_filename, tmp_img_with_bbox)

                else:
                    print(' Can not load image {}'.format(image_path))
            if 'video' in visualization_option:
                pathIn = result_dir
                pathOut = os.path.join(*[visualization_result_dir, 'video',
                                         dir_name + '.avi'])

                if self.image_to_video_fps == '15s':
                    num_frame = len(image_path_list)
                    fps = round(num_frame / 15.0)
                    if fps == 0: fps = 1
                    print(f'{len(image_path_list)} images, round(num_frame/15.0) = '
                          f'{round(num_frame / 15.0)}, fps = {fps} ')
                else:
                    fps = self.image_to_video_fps

                smrc.utils.convert_frames_to_video(pathIn, pathOut, fps)


# def visualize_bbox(image_dir, label_dir, class_list_file, resulting_visualization_dir):
#     ann_tool = AnnotationTool(image_dir=image_dir, label_dir=label_dir, class_list_file=class_list_file)
#
#     smrc.line.generate_dir_if_not_exist(resulting_visualization_dir)
#     image_path_list = smrc.line.get_file_list_in_directory(image_dir)
#     for i, image_path in enumerate(image_path_list):
#         ann_path = smrc.line.get_image_or_annotation_path(image_path, image_dir, label_dir, '.txt')
#         new_image_path = image_path.replace(image_dir, resulting_visualization_dir)
#
#         # visualization
#         tmp_img = cv2.imread(image_path)
#         ann_tool.draw_bboxes_from_file(tmp_img, ann_path)
#         cv2.imwrite(new_image_path, tmp_img)



