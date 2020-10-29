import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import random

from object_tracking.visualize import BBoxVisMode, Visualization
from object_tracking.data_hub import DataHub
import smrc.utils
from ._html import TrackHTML


class TrackingDebug(Visualization, TrackHTML):
    def __init__(self):
        super(Visualization, self).__init__()
        # we should not initialize TrackHTML
        # super(TrackHTML, self).__init__()

    def plot_cluster_connection(self, cluster, resulting_image_prefix=None):
        assert len(cluster) > 0

        # if len(cluster) > 1:
        #     count = 0
        #     for i in range(len(cluster) - 1):
        #         bbox_idx_prev, bbox_idx_next = cluster[i], cluster[i+1]
        #         resulting_image_name = 'clusters/round0_merge' + str(count) + '.jpg'
        #         self.connecting_two_bbox_visualization(
        #             bbox_idx_prev, bbox_idx_next,
        #             resulting_image_name
        #         )
        #         count += 1

        num = len(cluster)
        image_ids = self.get_image_id_list_for_cluster(cluster)
        for idx, global_bbox_id in enumerate(cluster):
            image_id = self.get_image_id(global_bbox_id)
            bbox = self.get_single_bbox(global_bbox_id)
            tmp_img = cv2.imread(self.IMAGE_PATH_LIST[image_id])
            text = f'image_id: {image_id}, {idx+1}/{num}, [{min(image_ids)}-{max(image_ids)}]\n ' \
                       f'bbox: [' + ' '.join(map(str, bbox)) + ']'
            self.draw_single_bbox_with_active_color_and_text(tmp_img, bbox, text)
            resulting_image_name = str(idx) + '_' + str(image_id) + '.jpg'
            if resulting_image_prefix is not None:
                resulting_image_name = resulting_image_prefix + resulting_image_name

            cv2.imwrite(resulting_image_name, tmp_img)

    def draw_single_bbox_with_active_color_and_text(self, tmp_img, bbox, text):
        self.draw_single_bbox(tmp_img, bbox, self.ACTIVE_BBOX_COLOR, self.ACTIVE_BBOX_COLOR, self.ACTIVE_BBOX_COLOR)
        # draw_single_bbox(self, tmp_img, bbox, rectangle_color, anchor_rect_color, text_shadow_color)

        smrc.utils.display_text_on_image(tmp_img, text, font_scale=2)

    def connecting_two_bbox_visualization(
            self, global_bbox_id1, global_bbox_id2, resulting_image_name
    ):
        """Visualizing the process of linking two detections.
        :param global_bbox_id1:
        :param global_bbox_id2:
        :param resulting_image_name:
        :return:
        """
        image_id1, image_id2 = self.get_image_id(global_bbox_id1), \
                               self.get_image_id(global_bbox_id2)
        img_name1, img_name2 = self.IMAGE_PATH_LIST[image_id1], \
                               self.IMAGE_PATH_LIST[image_id2]

        img1, img2 = cv2.imread(img_name1), cv2.imread(img_name2)
        bbox1, bbox2 = self.get_single_bbox(global_bbox_id1), \
                       self.get_single_bbox(global_bbox_id2)

        if img1 is None or img2 is None:
            print(f'{img_name1} or {img_name2} does not exist.')
            sys.exit(0)
        else:
            text1 = f'image_id: {image_id1}\n [' + ' '.join(map(str, bbox1)) + ']'
            self.draw_single_bbox_with_active_color_and_text(img1, bbox1, text1)

            text2 = f'image_id: {image_id2}\n [' + ' '.join(map(str, bbox2)) + ']'
            self.draw_single_bbox_with_active_color_and_text(img2, bbox2, text2)

            img_merged = smrc.utils.merging_two_images_side_by_side(img1, img2)
            print(f'Saving merged image to {resulting_image_name} ...')
            cv2.imwrite(resulting_image_name, img_merged)


        """
        input: video_detection_list, image list and their corresponding bbox for a single video 
            [ 
                [ image_name, bbox_list ]
            ]
            where bbox_list = [uncertain_class_label, x1, y1, x2, y2 ]
            if no masks for an image, bbox_list = []
        output:
            [ 
                [ image_name, bbox_list, object_id_list ]
            ]
            image_name, exactly the same with the input
            bbox_list, #may suffer from modification,
            object_id_list, 
                clustering results, each bbox is assigned an object id.
                same size with bbox_list

        How to maintain the connectedness ?
        """

    # Utilities for assistant the object_tracking process
    def plot_all_detected_bbox(self, result_dir='clusters/dets', video_path=None):
        """Draw images with detected bbox and generate videos if video_path is given
        :param result_dir:
        :return:
        """
        smrc.utils.generate_dir_if_not_exist(result_dir)
        for image_id, img_path in enumerate(self.IMAGE_PATH_LIST):
            # add key to the detected_bbox_list_dict
            tmp_img = cv2.imread(img_path)
            # detected_bbox = [self.get_single_bbox(x) for x in self.frame_dets[img_path]]

            img_path_new = img_path.replace(self.IMAGE_DIR, result_dir)

            for global_bbox_id in self.frame_dets[img_path]:
                # self.draw_detected_bbox_for_clustering(
                #     tmp_img, bbox
                # )
                bbox = self.get_single_bbox(global_bbox_id)
                score = self.get_detection_score(global_bbox_id)
                class_index, xmin, ymin, xmax, ymax = bbox

                cv2.rectangle(tmp_img, (xmin, ymin), (xmax, ymax), self.ACTIVE_BBOX_COLOR, 2)
                # draw resizing anchors
                # self.draw_bbox_anchors(tmp_img, xmin, ymin, xmax, ymax, self.ACTIVE_BBOX_COLOR, self.LINE_THICKNESS)

                text = "%.2f" % score
                if text is not None:
                    # class_name = text
                    # # text_shadow_color = class_color,
                    text_color = (0, 0, 0)  # , i.e., black
                    # int((ymin + ymax) / 2.0)
                    self.draw_class_name(tmp_img, (xmin, ymax + 25),
                                         text, self.ACTIVE_BBOX_COLOR,
                                         text_color)

            cv2.imwrite(filename=img_path_new, img=tmp_img)

        if video_path is not None:
            smrc.utils.convert_frames_to_video(pathIn=result_dir, pathOut=video_path, fps=30)

    def plot_all_clusters_on_blank_image_for_insight(
            self, clusters, num_sample_per_cluster=3
    ):
        h, w = smrc.utils.get_image_size(self.IMAGE_PATH_LIST[0])
        global_tmp_img = smrc.utils.generate_blank_image(height=h, width=w)
        image_dir_to_save = os.path.join('clusters', self._generate_image_dir_signature_prefix())
        smrc.utils.generate_dir_if_not_exist(image_dir_to_save)
        for i, cluster in enumerate(clusters):
            for j in range(num_sample_per_cluster):

                global_bbox_id = random.choice(cluster)
                image_id = self.get_image_id(global_bbox_id)
                bbox = self.get_single_bbox(global_bbox_id)

                image_name = self.IMAGE_PATH_LIST[image_id]
                tmp_img = cv2.imread(image_name)
                x1, y1, x2, y2 = bbox[1:5]
                roi = tmp_img[y1: y2, x1: x2, :]

                if j == 0:
                    global_tmp_img[y1: y2, x1: x2, :] = roi

                    cv2.rectangle(global_tmp_img, (x1, y1), (x2, y2), self.ACTIVE_BBOX_COLOR, self.LINE_THICKNESS)
                    # draw resizing anchors
                    self.draw_bbox_anchors(global_tmp_img, x1, y1, x2, y2, self.ACTIVE_BBOX_COLOR, self.LINE_THICKNESS)
                    text = 'obj' + str(i)
                    self.draw_class_name(global_tmp_img, (x1, y1), text, self.ACTIVE_BBOX_COLOR, self.BLACK)  #

                if j == 0:
                    new_image_name = os.path.join(image_dir_to_save, 'obj' + str(i) + '.jpg')
                    cv2.imwrite(new_image_name, roi)

                new_image_name = os.path.join(image_dir_to_save, 'obj' + str(i) + '_' + str(j) + '.jpg')
                cv2.imwrite(new_image_name, roi)


class TrackingDebugDeprecated(TrackingDebug):
    def __init__(self):
        super().__init__()

    def plot_all_clusters_on_blank_image_v3(self, clusters):
        h, w = smrc.utils.get_image_size(self.IMAGE_PATH_LIST[0])
        w *= 2
        global_tmp_img = smrc.utils.generate_blank_image(height=h, width=w)

        x_limit = w * 2
        y_limit = float('inf')
        margin = 20
        xmax, ymax = 0, 0
        ymax_list = []
        count = 0
        for i, cluster in enumerate(clusters):
            global_bbox_id = cluster[0]
            image_id = self.get_image_id(global_bbox_id)
            bbox = self.get_single_bbox(global_bbox_id)

            image_name = self.IMAGE_PATH_LIST[image_id]
            tmp_img = cv2.imread(image_name)
            x1, y1, x2, y2 = bbox[1:5]
            roi = tmp_img[y1: y2, x1: x2, :]
            if xmax + x2 - x1 + margin > w:
                xmax = 0
                ymax = max(ymax_list)
                ymax_list = []

            if ymax + y2 - y1 + margin > h * (count + 1):
                count += 1
                global_tmp_img_new = smrc.utils.generate_blank_image(height=h * (count + 1), width=w)
                tmp_h, tmp_w = global_tmp_img.shape[:2]
                global_tmp_img_new[:tmp_h, :tmp_w, :] = global_tmp_img
                global_tmp_img = global_tmp_img_new.copy()

            x_new, y_new = xmax + margin, ymax + margin
            x_shift, y_shift = x_new - x1, y_new - y1
            x1, y1, x2, y2 = (np.array([x1, y1, x2, y2]) + np.array([x_shift, y_shift, x_shift, y_shift])).tolist()

            global_tmp_img[y1: y2, x1: x2, :] = roi
            cv2.rectangle(global_tmp_img, (x1, y1), (x2, y2), self.ACTIVE_BBOX_COLOR, self.LINE_THICKNESS)
            # draw resizing anchors
            self.draw_bbox_anchors(global_tmp_img, x1, y1, x2, y2, self.ACTIVE_BBOX_COLOR, self.LINE_THICKNESS)
            text = 'obj' + str(i)
            self.draw_class_name(global_tmp_img, (x1, y1), text, self.ACTIVE_BBOX_COLOR, self.BLACK)  #
            new_image_name = 'clusters/all_cluster_plot.jpg'
            cv2.imwrite(new_image_name, global_tmp_img)

            xmax += x2 - x1 + margin
            # ymax += y2 - y1 + margin
            ymax_list.append(ymax + y2 - y1 + margin)
        # need to extend to image vertically
        new_image_name = 'clusters/all_cluster_plot.jpg'
        cv2.imwrite(new_image_name, global_tmp_img)

    def plot_all_clusters_on_blank_image_v0(self, clusters):
        # h, w = smrc.line.get_image_size(self.IMAGE_PATH_LIST[0])
        h, w = smrc.utils.get_image_size(self.IMAGE_PATH_LIST[0])
        global_tmp_img = smrc.utils.generate_blank_image(height=h, width=w)

        for i, cluster in enumerate(clusters):
            global_bbox_id = cluster[0]
            image_id = self.get_image_id(global_bbox_id)
            bbox = self.get_single_bbox(global_bbox_id)

            image_name = self.IMAGE_PATH_LIST[image_id]
            tmp_img = cv2.imread(image_name)
            x1, y1, x2, y2 = bbox[1:5]
            roi = tmp_img[y1: y2, x1: x2, :]
            global_tmp_img[y1: y2, x1: x2, :] = roi

            cv2.rectangle(global_tmp_img, (x1, y1), (x2, y2), self.ACTIVE_BBOX_COLOR, self.LINE_THICKNESS)
            # draw resizing anchors
            self.draw_bbox_anchors(global_tmp_img, x1, y1, x2, y2, self.ACTIVE_BBOX_COLOR, self.LINE_THICKNESS)
            text = 'obj' + str(i)
            self.draw_class_name(global_tmp_img, (x1, y1), text, self.ACTIVE_BBOX_COLOR, self.BLACK)  #

        new_image_name = 'clusters/all_cluster_plot.jpg'
        cv2.imwrite(new_image_name, global_tmp_img)

    def plot_all_clusters_on_blank_image_v1(
            self, clusters, new_image_name=None
    ):

        tmp_clusters_remain = {}
        for i, cluster in enumerate(clusters):
            tmp_clusters_remain[i] = cluster

        if new_image_name is None:
            new_image_name = 'clusters/all_cluster_plot.jpg'

        def overlap_all_object_on_blank_image(clusters_to_process, plot_name):
            h, w = smrc.utils.get_image_size(self.IMAGE_PATH_LIST[0])
            global_tmp_img = smrc.utils.generate_blank_image(height=h, width=w)

            clusters_remain = {}
            for i, cluster in clusters_to_process.items():

                global_bbox_id = random.choice(cluster)
                image_id = self.get_image_id(global_bbox_id)
                bbox = self.get_single_bbox(global_bbox_id)

                image_name = self.IMAGE_PATH_LIST[image_id]
                tmp_img = cv2.imread(image_name)
                x1, y1, x2, y2 = bbox[1:5]
                roi = tmp_img[y1: y2, x1: x2, :]


                if np.min(global_tmp_img[y1: y2, x1: x2, :]) == 255:
                    global_tmp_img[y1: y2, x1: x2, :] = roi

                    cv2.rectangle(global_tmp_img, (x1, y1), (x2, y2), self.ACTIVE_BBOX_COLOR, self.LINE_THICKNESS)
                    # draw resizing anchors
                    self.draw_bbox_anchors(global_tmp_img, x1, y1, x2, y2, self.ACTIVE_BBOX_COLOR, self.LINE_THICKNESS)
                    text = 'obj' + str(i)
                    self.draw_class_name(global_tmp_img, (x1, y1), text, self.ACTIVE_BBOX_COLOR, self.BLACK)  #
                else:
                    clusters_remain[i] = cluster
            # new_image_name = 'clusters/all_cluster_plot.jpg'

            cv2.imwrite(plot_name, global_tmp_img)
            return clusters_remain

        round = 0
        img = None
        while len(tmp_clusters_remain) > 0:
            print(f'Round {round}, len(tmp_clusters_remain) = {len(tmp_clusters_remain)} ..., ')
            new_image_name_tmp = 'clusters/all_cluster_plot' + str(round) + '.jpg'
            tmp_clusters_remain = overlap_all_object_on_blank_image(tmp_clusters_remain, new_image_name_tmp)

            # if img is None:
            #     final_tmp_img =
            round += 1

    def record_single_cluster_for_debugging(
            self, clusters, cluster_id, prefix=None
    ):
        """
        plot the images with the bbox included for visual analysis
        generate a txt file to save all the related information of the cluster
        :param clusters:
        :return: nothing
        """
        print('-------------------------------------')
        print(f'Begin to display the information for cluster {cluster_id}/{len(clusters)}')
        cluster = clusters[cluster_id]
        print(f'cluster = {cluster}')
        num_bbox = len(cluster)
        print(f'Total number of bbox: {num_bbox}')

        if num_bbox == 0:
            sys.exit(0)

        # all the images have the same root_dir
        global_bbox_id = cluster[0]
        image_id = self.get_image_id(global_bbox_id)
        image_name = self.IMAGE_PATH_LIST[image_id]

        # the position of the last name for the image, e.g., 0001.jpg vs. 2407/0001.jpg
        last_name_pos = image_name.rfind(os.path.sep)
        root_dir = image_name[: last_name_pos].replace(self.IMAGE_DIR, 'clusters')
        smrc.utils.generate_dir_if_not_exist(root_dir)

        # write cluster information to txt file
        if prefix is not None:
            txt_file_name = os.path.join(root_dir, prefix + '_object_' + str(cluster_id) + '.txt')
        else:
            txt_file_name = os.path.join(root_dir, 'object_' + str(cluster_id) + '.txt')

        init_image_id = self.get_image_id(cluster[0])
        last_image_id = self.get_image_id(cluster[-1])
        with open(txt_file_name, 'w') as new_file:
            # print(f'cluster = {cluster}')
            sub_dir_suffix = 'object' + str(cluster_id) + '_' + str(len(cluster))
            sub_dir = os.path.join(root_dir, sub_dir_suffix)
            smrc.utils.generate_dir_if_not_exist(sub_dir)

            for index, global_bbox_id in enumerate(cluster):
                image_id = self.get_image_id(global_bbox_id)
                bbox = self.get_single_bbox(global_bbox_id)

                image_name = self.IMAGE_PATH_LIST[image_id]
                tmp_img = cv2.imread(image_name)
                if tmp_img is not None:
                    # # the data format should be int type, class_idx is 0-index.
                    # class_idx = bbox[0]
                    # # draw bbox
                    # class_color = self.CLASS_BGR_COLORS[class_idx].tolist()
                    self.draw_single_bbox(
                        tmp_img, bbox, self.ACTIVE_BBOX_COLOR, self.ACTIVE_BBOX_COLOR, self.ACTIVE_BBOX_COLOR)
                    # draw_single_bbox(self, tmp_img, bbox, rectangle_color, anchor_rect_color, text_shadow_color)

                    text = f'{prefix} object {cluster_id}: {index}/{num_bbox-1} \n'
                    text += f'Image {image_id} [{init_image_id}, {last_image_id}]'
                    smrc.utils.display_text_on_image(tmp_img, text)
                    if prefix is not None:
                        last_name = prefix + 'object' + str(cluster_id) + '_' + str(len(cluster)) + \
                                    '_' + str(index) + '_' + image_name[last_name_pos + 1:]
                    else:
                        last_name = 'object' + str(cluster_id) + '_' + str(len(cluster)) + \
                                    '_' + str(index) + '_' + image_name[last_name_pos + 1:]

                    new_image_name = os.path.join(sub_dir, last_name)
                    print(f'new_image_name={new_image_name}')
                    cv2.imwrite(new_image_name, tmp_img)

                    # copy the first image outside
                    if index == 0:
                        cv2.imwrite(os.path.join(root_dir, last_name), tmp_img)

                    class_idx, x1, y1, x2, y2 = bbox
                    txt_line = f'{index}/{num_bbox-1}, image_id = {image_id}, class id = {class_idx}, [{x1}, {y1}, {x2}, {y2}]'
                    new_file.write(txt_line + '\n')

                else:
                    print(f'Image {image_name} does not exist, please check ...')
                    sys.exit(0)
        new_file.close()
