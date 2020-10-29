from tqdm import tqdm
import os
import cv2

from object_tracking.data_hub import DataHub
from object_tracking.visualize import Visualization

import smrc.utils


class TrackerSMRC(DataHub):
    def __init__(self):
        super().__init__()
        # initial default tracker params
        # The parameters for each method should be specified separately
        self.Tracker_Params = {}

    def _display_tracking_params(self):
        print('========================== Displaying tracking parameter:')
        if len(self.Tracker_Params) > 0:
            for key, value in self.Tracker_Params.items():
                print(f'param: {key}, value: {value} ...')
        print('Display done ==========================:')

    def init_tracking_params(self, **kwargs):
        if kwargs is not None:
            self.Tracker_Params.update(kwargs)
            self._display_tracking_params()

    # def load_video_detection_all(self, video_detection_list):
    #     """load the detection list for offline object_tracking
    #     initialize self.video_detected_bbox_all, self.frame_dets, self.connectedness
    #     :param video_detection_list: a list of detection list, each detection has the format
    #         of [class_idx, x1, y1, x2, y2, score] or [class_idx, x1, y1, x2, y2]
    #     :return:
    #     """
    #     self.video_detected_bbox_all = {}
    #     self.frame_dets = {}  # a point represent a list [image_id, bbox, bbox_id]
    #     self.IMAGE_PATH_LIST = []  # for quick access the image path
    #
    #     for image_id, img_detection in enumerate(video_detection_list):
    #         # print(img_detection)
    #         img_path, detection_list = img_detection
    #
    #         # load image path and bbox_list
    #         self.IMAGE_PATH_LIST.append(img_path)
    #         if len(detection_list) > 0:
    #             id_str = len(self.video_detected_bbox_all)
    #             for detection in detection_list:
    #                 global_bbox_id = len(self.video_detected_bbox_all)
    #                 if len(detection) == 5:  # no detection score [class_idx, x1, y1, x2, y2]
    #                     bbox = list(map(int, detection))
    #                     # self.video_detected_bbox_all[global_bbox_id] = Detection(image_id=image_id, bbox=bbox,
    #                     #                                                          score=1.0)
    #                     self.video_detected_bbox_all[global_bbox_id] = {
    #                         'image_id': image_id,
    #                         'bbox': bbox,
    #                         'score': 1.0  # we assume this detection is 100% confident
    #                     }
    #
    #                 elif len(detection) == 6:  # with detection score [class_idx, x1, y1, x2, y2, score]
    #                     bbox, score = list(map(int, detection[0:5])), detection[5]
    #                     # self.video_detected_bbox_all[global_bbox_id] = \
    #                     #     Detection(image_id=image_id, bbox=bbox, score=score)
    #                     self.video_detected_bbox_all[global_bbox_id] = {
    #                         'image_id': image_id,
    #                         'bbox': bbox,
    #                         'score': score  # we assume this detection is 100% confident
    #                     }
    #
    #             # only save the key of the detections
    #             self.frame_dets[img_path] = list(
    #                 range(id_str, len(self.video_detected_bbox_all))
    #             )
    #         else:
    #             self.frame_dets[img_path] = []
    #
    #     # self.LAST_IMAGE_INDEX = len(self.IMAGE_PATH_LIST) - 1
    #     print('Offline object_tracking: load detection list done.'
    #           '\nTotal number of detected bbox = %d '
    #           % (len(self.video_detected_bbox_all),))

    def save_tracking_results_as_txt_file(
            self, result_file_name,
            sorting_axis='track_length'  # 'track_length' 'image_id'
    ):
        assert len(self.clusters) > 0, 'There should be more than one cluster ...'
        available_sorting_axis = ['image_id', 'track_length']
        assert sorting_axis in available_sorting_axis, \
            f'{sorting_axis} not defined in {available_sorting_axis}. '
        print(f'Sorting clusters based on  {sorting_axis} ...')

        # to keep self.clusters as it is
        clusters = self.clusters.copy()
        if sorting_axis == 'image_id':
            clusters = self.sorted_clusters_based_on_image_id(clusters)
        else:
            clusters = self.sorted_clusters_based_on_length(clusters)
        print(f'Saving object_tracking results to {os.path.abspath(result_file_name)}:'
              f' Total {len(self.clusters)} objects  ...')

        with open(result_file_name, 'w') as new_file:
            for idx, cluster in enumerate(clusters):
                print(f'|   cluster {idx}, {len(cluster)} bboxes.')
                for global_bbox_id in cluster:
                    image_id = self.get_image_id(global_bbox_id)
                    bbox = self.get_single_bbox(global_bbox_id)
                    # image ID, object ID, class ID, x1, y1, x2, y2
                    result = [image_id, idx] + bbox
                    txt_line = ', '.join(map(str, result))
                    new_file.write(txt_line + '\n')

    def generate_video_for_tracking_result(
            self, clusters, resulting_dir,
            blank_bg=False,
            fps=30
    ):
        tracker_name = self.__class__.__name__

        print(f'Saving object_tracking results to {os.path.abspath(resulting_dir)} ...')
        smrc.utils.generate_dir_if_not_exist(resulting_dir)

        assert len(clusters) > 0, 'There should be more than one cluster ...'
        bbox_to_plot = [[] for image_id, _ in enumerate(self.IMAGE_PATH_LIST)]
        for cluster_id, cluster in enumerate(clusters):
            for global_bbox_id in cluster:
                image_id, bbox = self.get_image_id_and_bbox(global_bbox_id)
                bbox_to_plot[image_id].append([cluster_id, bbox])

        height, width = smrc.utils.get_image_size(
            self.IMAGE_PATH_LIST[0]
        )

        if width > 800:
            line_thickness = 2
            font_scale = 0.8
        else:
            line_thickness = 1
            font_scale = 0.6

        object_colors = smrc.utils.color.unique_colors(len(clusters))

        for image_id, image_path in tqdm(enumerate(self.IMAGE_PATH_LIST)):
            if blank_bg:
                tmp_img = smrc.utils.generate_blank_image(height, width)
            else:
                tmp_img = cv2.imread(image_path)
            # display the tracker name on the image
            smrc.utils.draw.put_text_on_image(
                tmp_img, text_content=tracker_name,
                thickness=4, font_scale=2
            )
            # plot tracking result if any
            if len(bbox_to_plot[image_id]) > 0:
                for object_id, bbox in bbox_to_plot[image_id]:
                    _, xmin, ymin, xmax, ymax = bbox

                    object_color = object_colors[object_id]
                    text = 'obj ' + str(object_id)
                    smrc.utils.draw.draw_bbox_legend(
                        tmp_img=tmp_img, text_content=text,
                        location_to_draw=(xmin, ymin),
                        text_shadow_color=object_color,
                        text_color=(0, 0, 0),
                        font_scale=font_scale,
                        line_thickness=line_thickness
                    )
                    cv2.rectangle(
                        tmp_img, (xmin, ymin), (xmax, ymax),
                        object_color, line_thickness
                    )

            image_path_new = os.path.join(
                    resulting_dir, os.path.basename(image_path)
            )
            # print(f'Saving object_tracking results to {image_path_new}')
            cv2.imwrite(image_path_new, tmp_img)

        pathIn, pathOut = resulting_dir, resulting_dir + '.avi'
        print(f'Generating object_tracking results to {os.path.abspath(pathOut)} ...')
        smrc.utils.convert_frames_to_video(pathIn, pathOut, fps)

    def preprocessing(self, min_detection_score=None, nms_thd=None, **kwargs):
        """Mark a part of detections as false positives, e.g., detections with
        low score or suppressed detections by nms
        Later, we may recycle the 'false positives'.
        conduct nms and remove low detections scores if they are specified.
        :param min_detection_score:
        :param nms_thd:
        :param kwargs:
        :return:
        """

        from smrc.utils.det.detection_process import non_max_suppression_selected_index
        false_positives = []
        if nms_thd is not None:
            count = 0
            for image_path in self.IMAGE_PATH_LIST:
                global_bbox_list = self.frame_dets[image_path]
                if len(global_bbox_list) > 0:
                    image_pred = [self.get_single_bbox(x) + [self.get_detection_score(x)]
                                  for x in global_bbox_list]
                    remained_index = non_max_suppression_selected_index(
                        image_pred=image_pred, nms_thd=nms_thd, score_position='last'
                    )
                    false_positive_index = set(range(len(image_pred))) - set(remained_index)
                    global_bbox_id_to_remove_list = [global_bbox_list[i] for i in false_positive_index]

                    for global_bbox_id in global_bbox_id_to_remove_list:
                        # self.frame_dets[image_path].remove(global_bbox_id)
                        false_positives.append(global_bbox_id)
                        count += 1

            print(f'{count} detections removed due to nms thd: {nms_thd} ...')

        if min_detection_score is not None:
            count = 0
            for image_path in self.IMAGE_PATH_LIST:
                global_bbox_id_to_remove_list = []

                global_bbox_list = self.frame_dets[image_path]
                for global_bbox_id in global_bbox_list:
                    if self.get_detection_score(global_bbox_id) < min_detection_score:
                        global_bbox_id_to_remove_list.append(global_bbox_id)

                for global_bbox_id in global_bbox_id_to_remove_list:
                    # self.frame_dets[image_path].remove(global_bbox_id)
                    false_positives.append(global_bbox_id)
                    # # del clusters[global_bbox_id]
                    count += 1
            print(f'{count} detections removed due to min_detection_score: {min_detection_score} ...')
        false_positives = list(set(false_positives))
        print(f'{len(false_positives)} detections are regarded as false positives with'
              f' min_detection_score={min_detection_score}, nms_thd={nms_thd}')
        print(f'------------------------------------------------')
        print('Note: If you want to remove false_positives by preprocessing, please '
              'explicitly call self._move_det_to_false_positives(). Currently, not removed yet.')
        print(f'------------------------------------------------')
        return false_positives

    def init_tracking_tool(self, video_detection_list, **kwargs):
        self.clusters = []
        # the detection data and image list are re-initialized in load_detected_bbox_all
        # transfer detection to the object_tracking data format
        self.load_video_detection_all(
            video_detection_list=video_detection_list
        )

        # if any settings for the method is given, than update the setting
        self.init_tracking_params(**kwargs)

        # if self.gtruth is not None and len(self.video_detected_bbox_all) > 0:
        #     self.build_correspondence_det_and_gtruth()

    def offline_tracking(self, video_detection_list, **kwargs):
        pass

    def visualize_tracking_results(self):
        visualizer = Visualization()
        # visualizer.visualization_mode = BBoxVisMode.ObjectID
        visualizer.from_tracker(self)
        visualizer.visualize_tracking_results()

