##################################################
# visualizing the tracking results with html format
##################################################
import os
import webbrowser
import numpy as np
# import random
import cv2
# import matplotlib.pyplot as plt
from tqdm import tqdm

from object_tracking.data_hub import DataHub
import smrc.utils


# inf_value = float("inf")


class TrackHTML(DataHub):
    def __init__(
            self,
            seq_dir, det_or_gt='det',
            video_detection_list=None,
            video_detected_bbox_all=None,
            data_root_dir=''
            ):
        super(DataHub, self).__init__()
        self.DATA_ROOT_DIR = data_root_dir
        self.RELATIVE_SEQ_DIR = seq_dir
        # set where to save the roi images

        # 'clusters/html_images',  we put the html in the same dir under 'clusters/html_images',
        self.html_root_dir = os.path.join(self.DATA_ROOT_DIR, 'clusters')

        # generate the directories if not exist
        smrc.utils.generate_dir_if_not_exist(self.html_root_dir)

        assert video_detected_bbox_all is not None or video_detection_list is not None
        if video_detected_bbox_all is not None:
            self.video_detected_bbox_all = video_detected_bbox_all
        elif video_detection_list is not None:
            self.load_video_detection_all(video_detection_list)

        self.det_or_gt = det_or_gt

    def is_roi_image_ready(self):
        dir_path = self._get_full_det_dir_for_cur_sequence()
        if not os.path.isdir(dir_path):
            return False
        else:
            image_path_list = smrc.utils.get_file_list_recursively(dir_path)
            return len(image_path_list) == len(self.video_detected_bbox_all)

    def _get_full_det_dir_for_cur_sequence(self):
        return os.path.join(
            self.html_root_dir, self.det_or_gt,
            self._generate_image_dir_signature_prefix()
        )

    # def _get_relative_det_dir_for_html_use(self):
    #     """Using relative path is preferred for html file generation
    #     (due to simplicity and transferability as it is independent to computer path).
    #     :return:
    #     """
    #     return os.path.join(self.det_or_gt, self._generate_image_dir_signature_prefix())

    def _generate_image_dir_signature_prefix(self):
        return self.RELATIVE_SEQ_DIR.replace(os.path.sep, '_')

    def visualize_clusters_html(
            self, clusters, title=None,
            filename_signature=None, open_file=False
    ):
        assert len(clusters) > 0
        clusters = self.sorted_clusters_based_on_length(clusters)
        # self.plot_all_cluster_roi_on_separate_folders(clusters)
        if filename_signature is None:
            filename_signature = \
                self._generate_image_dir_signature_prefix() + '_resulting_clusters'
        # if self.det_or_gt == 'det':
        #     filename = os.path.join(self.html_root_dir, filename_signature + '.html')
        # else:
        filename = os.path.join(self.html_root_dir, self.det_or_gt, filename_signature + '.html')
        self.generate_resulting_clusters_html_file(
            clusters=clusters, filename=filename, title=title, open_file=open_file
        )

    def generate_resulting_clusters_html_file(
            self, clusters, filename, resize=True, title=None, open_file=False
    ):
        """generate html file for resulting clusters
        :param clusters:
        :param filename:
        :param resize:
        :param title:
        :param open_file:
        :return:
        """
        # size = "12", size = "8",
        if filename is None:
            filename = 'resulting_clusters' + smrc.utils.time_stamp_str() + '.html'
        f = open(filename, 'w')

        message = f"""<html>
<head></head>
<body>
<h1>{title}</h1>
<table border="1" align = 'center'>
"""
        obj_height, obj_width = 100, 40

        # if self.html_detection_image_dir is None:
        #     image_dir_name = 'clusters/html_images/detections'
        # else:
        # self._get_relative_det_dir_for_html_use()
        image_dir_name = self._generate_image_dir_signature_prefix()  # 'clusters/html_images/detections'

        for i, cluster in enumerate(clusters):
            message += f"""<tr>
    <td>
    <font color = "red">cluster {i} (len: {len(cluster)})</font>
    </td>
            """

            # generate table for one cluster
            message += f"""
    <td><table border="1" align = 'left' ><tr>"""
            for j, global_bbox_id in enumerate(cluster):
                # src_image_path = os.path.join(image_dir_name, 'obj' + str(i) + '_' + str(j) + '.jpg')
                src_image_path = os.path.join(image_dir_name, str(global_bbox_id) + '.jpg')
                if resize:
                    message += f"""
            <td><img src="{src_image_path}" height = "{obj_height}", width = "{obj_width}"></br>
{self.get_image_id(global_bbox_id)}</td>"""
                else:
                    message += f"""
                    <td><img src="{src_image_path}"></br>{self.get_image_id(global_bbox_id)}</td>"""

                if (j+1) % 50 == 0 and j > 0 and j != len(cluster)-1:
                    message += f"""</tr><tr>"""
            message += f"""</tr></table></td>"""

            message += "</tr>"
        message += "</table>"
        # appearance_dist_matrix[spatial_dist_matrix == float('inf')] = -1

        message += """\n</body>
        </html>"""

        f.write(message)
        f.close()
        if open_file:
            webbrowser.open_new_tab(filename)

    def generate_html_file_for_distance_matrix(
            self, filename, distance_matrix, top_k=10, resize=True, title=''
    ):
        """generate html file for distance matrix. We need to refactor this function to
        generate_html_file_for_distance_matrix
        :param filename:
        :param distance_matrix:
        :param top_k:
        :param resize:
        :param title:
        :return:
        """
        #
        # size = "12", size = "8",
        image_dir_to_save = os.path.join('clusters', self._generate_image_dir_signature_prefix())

        f = open(filename, 'w')

        message = f"""<html>
         <head></head>
         <body>
         <h1>{title}</h1>
         <table border="1" align = 'center' >
         """
        obj_height, obj_width = 100, 40
        num_row = distance_matrix.shape[0]

        def write_image_to_tabular(html_message, src_image_path, resize_option, img_height, img_width):
            if resize_option:
                html_message += f"""
                 <img src="{src_image_path}" height = "{img_height}", width = "{img_width}" ></br>"""
            else:
                html_message += f"""
                 <img src="{src_image_path}"></br>"""

            return html_message

        def write_major_object_image_to_tabular(
                html_message, src_image_path_list_, resize_option,
                img_height, img_width
        ):
            html_message += f"""<table><tr>"""
            for src_image_path in src_image_path_list_:
                if resize_option:
                    html_message += f"""
                     <td><img src="{src_image_path}" height = "{img_height}", width = "{img_width}"></td>"""
                else:
                    html_message += f"""
                     <td><img src="{src_image_path}"></td>"""

            html_message += f"""</tr></table></br>"""

            return html_message

        num_sample_per_cluster = 3
        if num_row > top_k:
            for i in range(num_row):
                # image_path = 'clusters/obj' + str(i) + '.jpg'
                # generate multiple object image list
                src_image_path_list = [os.path.join(image_dir_to_save, 'obj' + str(i) + '_' + str(j) + '.jpg')
                                       for j in range(num_sample_per_cluster)]
                # , height = "400", width = "640"
                message += f"""
                 <tr>
                 <td>
                 """
                message = write_major_object_image_to_tabular(
                    message, src_image_path_list_=src_image_path_list, resize_option=resize,
                    img_height=obj_height, img_width=obj_width
                )

                message += f"""
                     <font color = "red">cluster {i} (len: {len(self.clusters[i])})</font>
                     </td>
                 """
                # """<tr><th>Object</th><th>Good</th><th>Bad</th><th>Ugly</th></tr>"""

                # for fruit in d:
                #     html += "<tr><td>{}</td>".format(fruit)
                #     for state in "good", "bad", "ugly":
                #         html += "<td>{}</td>".format('<br>'.join(f for f in d[fruit] if ".{}.".format(state) in f))
                #     html += "</tr>"
                # html += "</table></html>"

                dist_list = distance_matrix[i, :]
                sort_index = np.argsort(dist_list)  # return increasing order, array()

                last_top_k = list(sort_index[-top_k - 1:])
                last_top_k_cleaned = [x for x in last_top_k if x != i]
                last_top_k = last_top_k_cleaned[-top_k:]

                for ind, j in enumerate(list(sort_index[:top_k]) + last_top_k):
                    similar_image_path = os.path.join(image_dir_to_save, 'obj' + str(j) + '.jpg')
                    message += f"""
                     <td>
                     """
                    message = write_image_to_tabular(
                        message, src_image_path=similar_image_path, resize_option=resize,
                        img_height=obj_height, img_width=obj_width
                    )
                    # dist:
                    if ind >= top_k:
                        message += f"""
                         C{j} </br> (<font color = "blue">{"%.1f" % distance_matrix[i, j]})</font>
                         </td>
                         """
                    else:
                        message += f"""
                         C{j} </br> (<font color = "green">{"%.2f" % distance_matrix[i, j]})</font>
                         </td>
                         """
                    # distance_matrix[i, sort_index[top_k:]] = float('inf')

                message += "</tr>"
        else:
            print(f'Attention please: num_row = {num_row}, top_k = {top_k}')
        message += "</table>"
        # appearance_dist_matrix[spatial_dist_matrix == float('inf')] = -1

        message += """
         </body>
         </html>"""

        f.write(message)
        f.close()

        webbrowser.open_new_tab(filename)

    def plot_all_detection_roi_for_html_visualization(
            self, result_dir=None):
        """Plot each roi of the detections into images for html visualization
        :param result_dir:
        :return:
        """
        if result_dir is None:
            result_dir = self._get_full_det_dir_for_cur_sequence()
        smrc.utils.generate_dir_if_not_exist(result_dir)

        # for i, detection in enumerate(self.video_detected_bbox_all):
        #     image_id, bbox, global_bbox_id = detection

        global_bbox_id_list = sorted(list(self.video_detected_bbox_all.keys()))
        pbar = tqdm(global_bbox_id_list)
        for global_bbox_id in pbar:
            image_id, bbox = self.get_image_id_and_bbox(global_bbox_id)
            bbox = [int(x) for x in bbox]
            pbar.set_description(
                f'Plotting detection for html visualization '
                f'{global_bbox_id}/{len(self.video_detected_bbox_all)}, bbox = {bbox} ... ')

            image_name = self.IMAGE_PATH_LIST[image_id]
            tmp_img = cv2.imread(image_name)
            img_h, img_w = tmp_img.shape[:2]
            x1, y1, x2, y2 = bbox[1:5]
            # it will cause errors for extracting roi if x1 < 0, e.g., -8,
            x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, img_w), min(y2, img_h)

            roi = tmp_img[y1: y2, x1: x2, :]
            new_image_name = os.path.join(result_dir, str(global_bbox_id) + '.jpg')
            cv2.imwrite(new_image_name, roi)

    def plot_all_cluster_roi_on_separate_folders(
            self, clusters, root_dir_name='clusters/html_images'
    ):
        smrc.utils.generate_dir_if_not_exist(root_dir_name)
        for i, cluster in enumerate(clusters):
            cluster = self.sort_cluster_based_on_image_id(cluster)

            for j, global_bbox_id in enumerate(cluster):
                image_id = self.get_image_id(global_bbox_id)
                bbox = self.get_single_bbox(global_bbox_id)

                image_name = self.IMAGE_PATH_LIST[image_id]
                tmp_img = cv2.imread(image_name)
                x1, y1, x2, y2 = bbox[1:5]
                roi = tmp_img[y1: y2, x1: x2, :]

                # if j == 0:
                #     global_tmp_img[y1: y2, x1: x2, :] = roi
                #     cv2.rectangle(global_tmp_img, (x1, y1), (x2, y2), self.ACTIVE_BBOX_COLOR, self.LINE_THICKNESS)
                #     # draw resizing anchors
                #     self.draw_bbox_anchors(global_tmp_img, x1, y1, x2, y2, self.ACTIVE_BBOX_COLOR, self.LINE_THICKNESS)
                #     text = 'obj' + str(i)
                #     self.draw_class_name(global_tmp_img, (x1, y1), text, self.ACTIVE_BBOX_COLOR, self.BLACK)  #
                #     new_image_name = os.path.join(dir_name, 'obj' + str(i) + '.jpg'
                #     cv2.imwrite(new_image_name, roi)

                # new_image_name = 'clusters/obj' + str(i) + '_' + str(j) + '.jpg'
                new_image_name = os.path.join(root_dir_name, 'obj' + str(i) + '_' + str(j) + '.jpg')
                cv2.imwrite(new_image_name, roi)

