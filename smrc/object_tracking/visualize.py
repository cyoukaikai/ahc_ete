import numpy as np
import cv2
import os
from tqdm import tqdm
# from time import sleep

from object_tracking.data_hub import DataHub
from smrc.utils.annotate.annotation_tool import AnnotationTool
import smrc.utils


class BBoxVisMode:
    """Enumerate type for the bbox.
    """
    ClassLabel = 0  # display the  class label
    BBoxRect = 1  # trajectory, only the rectangle of the object
    ObjectID = 2  # display the object ID
    ObjectPlusClass = 3  # display the object and class ID


class Visualization(AnnotationTool, DataHub):
    def __init__(self, class_list_file=None):
        super(DataHub).__init__()
        # super(AnnotationTool).__init__(class_list_file=class_list_file)
        # DataHub.__init__(self)
        AnnotationTool.__init__(
            self, class_list_file=class_list_file
        )
        ''' visualization section '''
        self.IMAGE_WINDOW_NAME = 'VisualizeTrackingResults'
        self.TRACKBAR_IMG = 'Image'
        self.TRACKBAR_CLUSTER = 'Object'

        # this should be False, because this mode conflicts with the default image view mode
        self.show_active_cluster_on = False
        self.active_cluster_id = None
        self.active_cluster_id_trackbar = 0

        self.LINE_THICKNESS = 1
        self.class_name_font_scale = 0.6

        # set how to display the bbox when visualizing the object_tracking results
        self.visualization_mode = BBoxVisMode.ObjectID
        self.object_colors = []
        self.object_id_to_display = {}

        # clusters trajectory
        self.clusters_trajectory = []
        self.cluster_labels = []

        # below deprecated
        # # the feature points in the old image frame
        # self.old_corners = {}  # key, image_path
        # # the feature points tracked in the new image frame
        # self.new_corners = {}
        #
        # self.connectivity = {}
        # self.show_merging_process = False  # debug purpose

    def from_tracker(self, tracker):
        self.video_detected_bbox_all = tracker.video_detected_bbox_all
        self.clusters = tracker.clusters
        self.IMAGE_PATH_LIST = tracker.IMAGE_PATH_LIST
        self.LAST_IMAGE_INDEX = len(self.IMAGE_PATH_LIST) - 1

        # no need recover_frame_dets, as we do not allow the tracker to modify self.frame_dets
        # self.recover_frame_dets()
        self.frame_dets = tracker.frame_dets

        # self.bbox_cluster_IDs = {}  # {}
        # self.frame_dets may have been modified by the traker
        # self.frame_dets = tracker.frame_dets

        # self.visualize_tracking_results()

    def visualize_tracking_results(self, additional_drawing_flag=None):
        """visualize object_tracking results for single directory
        :param additional_drawing_flag: 'IoU' or 'EuclideanDistance'
            or 'OpticalFlowIoU', or 'OpticalFlowPointCount'
        :return:
        """

        self.post_process_after_offline_tracking()

        # for directory {}'.format(self.active_directory )
        print('Start visualize the object_tracking results ...')
        # print('self.CLASS_LIST = ', self.CLASS_LIST)

        # this function must be after self.load_image_sequence(), otherwise, the trackBar
        # for image list can not be initialized (as number of image not known)
        self.init_image_window_and_mouse_listener()

        # self.load_image_list_and_detected_bbox()
        # print(self.IMAGE_PATH_LIST)
        # load the first image in the IMAGE_PATH_LIST,
        # initilize self.active_image_index and related information for the active image
        self.set_image_index(0)

        while True:
            # load the class index and class color for plot
            # color = self.CLASS_BGR_COLORS[self.active_class_index].tolist()

            # copy the current image
            tmp_img = self.active_image.copy()
            #
            # if self.show_active_cluster_on:
            #     text_content = 'Press c to show all the object_tracking result. \n'
            #     text_content += f'Object {self.active_cluster_id_trackbar}, ' \
            #         f'{len(self.clusters[self.active_cluster_id])} detection'
            #
            #     smrc.line.display_text_on_image(tmp_img, text_content)
            # else:
            #     smrc.line.display_text_on_image(tmp_img, 'Press c to show a single object.')

            image_path = self.IMAGE_PATH_LIST[self.active_image_index]
            # image_name = img_path.split(os.path.sep)[-1]
            self.display_additional_infor(tmp_img)

            self.visualize_clustered_object_on_active_image(image_path, tmp_img, additional_drawing_flag)

            # set the active directory based on mouse cursor
            self.set_active_bbox_idx_based_on_mouse_position()
            self.draw_active_bbox(tmp_img)

            cv2.imshow(self.IMAGE_WINDOW_NAME, tmp_img)

            ''' Key Listeners START '''
            pressed_key = self.read_pressed_key()

            # 255 Linux or Windows, cv2.waitKeyEx() & 0xFF , -1  Windows cv2.waitKeyEx() or Linux  cv2.waitKey(), 0 Windows cv2.waitKey()
            if pressed_key != 255 and pressed_key != 0 and pressed_key != -1:
                # print('pressed_key=', pressed_key)  # ('pressed_key=', -1) if no key is pressed.
                # print('pressed_key & 0xFF =', pressed_key & 0xFF)
                # print('self.platform = ', self.platform)
                # handle string key a -z
                if ord('a') <= pressed_key <= ord('z'):  # 'a': 97, 'z': 122
                    if pressed_key == ord('a') or pressed_key == ord('d'):
                        # show previous image key listener
                        if pressed_key == ord('a'):
                            self.active_image_index = smrc.utils.decrease_index(self.active_image_index,
                                                                                self.LAST_IMAGE_INDEX)
                        # show next image key listener
                        elif pressed_key == ord('d'):
                            self.active_image_index = smrc.utils.increase_index(self.active_image_index,
                                                                                self.LAST_IMAGE_INDEX)
                        cv2.setTrackbarPos(self.TRACKBAR_IMG, self.IMAGE_WINDOW_NAME, self.active_image_index)

                    elif (pressed_key == ord('j') or pressed_key == ord('k')) and self.show_active_cluster_on and len(
                            self.clusters) > 1:
                        if pressed_key == ord('j'):
                            self.active_cluster_id_trackbar = smrc.utils.decrease_index(
                                self.active_cluster_id_trackbar,
                                len(self.clusters) - 1
                            )
                        # show next image key listener
                        elif pressed_key == ord('k'):
                            self.active_cluster_id_trackbar = smrc.utils.increase_index(
                                self.active_cluster_id_trackbar,
                                len(self.clusters) - 1
                            )

                        cv2.setTrackbarPos(self.TRACKBAR_CLUSTER, self.IMAGE_WINDOW_NAME,
                                           self.active_cluster_id_trackbar)
                    elif pressed_key == ord('c'):
                        if not self.show_active_cluster_on:
                            self.show_active_cluster_on = True
                            #  -1 if not exist
                            checkTrackBarPos = cv2.getTrackbarPos(self.TRACKBAR_CLUSTER, self.IMAGE_WINDOW_NAME)

                            # never put this in the while loop, otherwise, error 'tuple object
                            # is not callable' (probably multiple createTrackbar generated)
                            if self.show_active_cluster_on and len(self.clusters) > 1 and checkTrackBarPos == -1:
                                cv2.createTrackbar(self.TRACKBAR_CLUSTER, self.IMAGE_WINDOW_NAME,
                                                   0, len(self.clusters) - 1,
                                                   self.set_cluster_id)

                            # begin to view the first cluster
                            self.set_cluster_id(0)
                        else:
                            self.show_active_cluster_on = False

                            cv2.destroyWindow(self.IMAGE_WINDOW_NAME)
                            self.init_image_window_and_mouse_listener()
                            cv2.setTrackbarPos(self.TRACKBAR_IMG, self.IMAGE_WINDOW_NAME, self.active_image_index)

                    # show edges key listener
                    elif pressed_key == ord('e'):
                        if self.show_tracked_corners_on:
                            self.show_tracked_corners_on = False
                            self.display_text('Show tracked features OFF!', 1000)
                        else:
                            self.show_tracked_corners_on = True
                            self.display_text('Show tracked features On!', 1000)

                elif pressed_key & 0xFF == 27:  # Esc key is pressed
                    # close the window
                    cv2.destroyWindow(self.IMAGE_WINDOW_NAME)
                    break

            if self.WITH_QT:
                # if window gets closed then quit
                if cv2.getWindowProperty(self.IMAGE_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    # close the window
                    cv2.destroyWindow(self.IMAGE_WINDOW_NAME)
                    break

        # if quit the annotation tool, close all windows
        cv2.destroyAllWindows()

    def post_process_after_offline_tracking(self):
        # self.cluster_sorted_index = sorted(range(len(self.clusters)),
        #                                    key=lambda k: len(self.clusters[k]),
        #                                    reverse=True)  # sort the list based on its first element
        #
        # # # sort clusters based on the minimum frame id by increasing value
        # # self.cluster_sorted_index = sorted(range(len(self.clusters)),
        # #                                    key=lambda k: min(self.get_image_id_list_for_cluster(self.clusters[k])),
        # #                                    reverse=False)  # sort the list based on its first element
        #
        # self.cluster_labels = self.estimate_cluster_label(self.clusters)

        # # initialize the object id for all the detections for quick access the
        # # object id given the global bbox id
        self._assign_cluster_id_to_global_bbox_idx()

        if self.visualization_mode == BBoxVisMode.ObjectPlusClass and \
                self.CLASS_LIST is not None and len(self.CLASS_LIST) > 1:
            self.estimate_display_object_id(self.CLASS_LIST)
        else:
            self.init_object_colors()

    def display_additional_infor(self, tmp_img):
        image_path = self.IMAGE_PATH_LIST[self.active_image_index]
        image_dir, video_name, image_name = smrc.utils.split_image_path(
                image_path
            )
        text_content = f'[{video_name}/{image_name}] '
        smrc.utils.display_text_on_image(tmp_img, text_content)

        if self.show_active_cluster_on:
            text_content += f'\ncluster {self.active_cluster_id}: {len(self.clusters[self.active_cluster_id])} detection'
            smrc.utils.display_text_on_image_top_middle(tmp_img, text_content, self.RED)

    def visualize_clustered_object_on_active_image(
            self, image_path, tmp_img, additional_drawing_flag=None
    ):
        """
        display the clustering results on active image in the GUI process
        :param image_path:
        :param tmp_img:
        :param additional_drawing_flag: 'IoU' or 'EuclideanDistance'
            or 'OpticalFlowIoU', or 'OpticalFlowPointCount'
        :return:
        """
        # connectedness = self.connectivity['connectedness'] if len(self.connectivity) > 0 else {}
        # self.active_image_detected_bboxes = []  # initialize the image_detected_bboxes

        # image_path = self.IMAGE_PATH_LIST[self.active_image_index]
        image_id = None
        for global_bbox_id in self.frame_dets[image_path]:
            image_id, bbox = self.get_image_id_and_bbox(global_bbox_id)
            # print(f'image_id={image_id}, bbox = {bbox}, bbox_id = {bbox_id}')

            # do not show the bbox is it is not included in a given cluster id.
            if self.show_active_cluster_on and (global_bbox_id not in self.clusters[self.active_cluster_id]):
                continue
            else:
                # append the bbox and it will be display later
                # self.active_image_detected_bboxes.append(bbox)

                if self.visualization_mode == BBoxVisMode.ObjectPlusClass:
                    corrected_class_idx, object_id_to_display = self.object_id_to_display[global_bbox_id]

                    # not assigned to any cluster, (regarded as outlier)
                    if object_id_to_display is not None:
                        self.draw_cluster_single_bbox_with_corrected_label(
                            tmp_img, bbox, corrected_class_idx, object_id_to_display
                        )
                elif self.visualization_mode == BBoxVisMode.ObjectID:
                    object_id = self.bbox_cluster_IDs[global_bbox_id]
                    self.draw_bbox(tmp_img, bbox, object_id=object_id)

        if image_id is None:
            image_id = self.IMAGE_PATH_LIST.index(image_path)
        self.draw_clustered_object_trajectory_online(tmp_img, cur_image_id=image_id)

    def draw_cluster_single_bbox_with_corrected_label(
            self, tmp_img, bbox, corrected_class_idx, object_id, text_color=(0, 0, 0)
    ):
        """
        We show the modified detection results (based on object_tracking), rather than show the original detection
        outlier, wrong detections will be deleted or modified during the process
        :param tmp_img: image
        :param bbox: original detection results
        :param corrected_class_idx:
        :param object_id:
        :param text_color:
        :return:
        """
        class_idx = corrected_class_idx
        bbox_rect = bbox[1:5]
        self.draw_cluster_single_bbox_rect(tmp_img, bbox_rect, class_idx, object_id, text_color)

    def draw_cluster_single_bbox_rect(
            self, tmp_img, bbox_rect, class_idx, object_id, text_color=(0, 0, 0)
    ):
        xmin, ymin, xmax, ymax = bbox_rect
        class_color = self.CLASS_BGR_COLORS[class_idx].tolist()
        class_name = self.CLASS_LIST[class_idx]
        text_name = class_name + ' ' + str(object_id)
        # text_shadow_color = class_color, text_color = (0, 0, 0), i.e., black
        self.draw_class_name(tmp_img, (xmin, ymin), text_name, class_color, text_color)  #
        cv2.rectangle(tmp_img, (xmin, ymin), (xmax, ymax), class_color, self.LINE_THICKNESS)

    def draw_bbox(self, tmp_img, bbox, object_id=None, draw_not_clustered_bbox=False):
        """
        Sub Function for visualize_clustered_object_on_active_image
        Draw bbox in the active image
        :param tmp_img:
        :param bbox:
        :param object_id:
        :param draw_not_clustered_bbox:
        :return:
        """
        _, xmin, ymin, xmax, ymax = bbox

        if object_id is None:
            if draw_not_clustered_bbox:
                object_color = self.RED
                text = 'deleted'
                # text_shadow_color = class_color, text_color = (0, 0, 0), i.e., black
                self.draw_class_name(tmp_img, (xmin, ymin), text, object_color, text_color=(0, 0, 0))  #
            else:
                return
        else:
            object_color = self.object_colors[object_id].tolist()
            text = 'obj ' + str(object_id)
            # text_shadow_color = class_color, text_color = (0, 0, 0), i.e., black
            # print(f'(xmin, ymin) = {(xmin, ymin)}')
            # print(f'text_content = {text}')
            # print(f'text_shadow_color = {object_color}')
            # print(f'text_color = {(0, 0, 0)}')

            self.draw_class_name(tmp_img, (xmin, ymin), text_content=text,
                                 text_shadow_color=object_color, text_color=(0, 0, 0))  #
            # self.draw_class_name(tmp_img, (xmin, ymin), text, (0, 0, 0), 1)  #
            # sys.exit(0)
        cv2.rectangle(tmp_img, (xmin, ymin), (xmax, ymax), object_color, self.LINE_THICKNESS)

    # Section: interactive visualization GUI
    def set_cluster_id(self, id):
        # transfer the sorted cluster id to the extact cluster (that are based on appearing time)
        # self.active_cluster_id = self.cluster_sorted_index[id]
        if id >= len(self.clusters):
            return

        self.active_cluster_id = id

        # print('The cluster id in track bar is %d , the active_cluster_id is %d '%(id, self.active_cluster_id))
        # to show the images in the specified cluster for visualization purpose
        # self.image_bbox_in_cluster = {}
        if 0 <= self.active_cluster_id < len(self.clusters):

            #show the images
            # bbox_ids = self.clusters[self.active_cluster_id]
            global_bbox_id_list_sorted = self.sort_cluster_based_on_image_id(
                self.clusters[self.active_cluster_id]
            )
            minimum_image_id = self.get_image_id(global_bbox_id_list_sorted[0])
            #
            # for bbox_id in bbox_ids:
            #     image_id, bbox, _ = self.video_detected_bbox_all[bbox_id]

            #
            #     # if self.IMAGE_PATH_LIST[image_id] not in self.image_bbox_in_cluster:
            #     #     self.image_bbox_in_cluster[ self.IMAGE_PATH_LIST[image_id] ] = []
            #     # self.image_bbox_in_cluster[ self.IMAGE_PATH_LIST[image_id] ].append(bbox)
            #
            #     if image_id < minimum_image_id:
            #         minimum_image_id = image_id

                    # print('The active_image_index id before modifying is %d  '%( self.active_image_index,))

            # redirect to the first image in the cluster
            #  set_image_index(minimum_image_id) and update the self.active_image_index
            cv2.setTrackbarPos(self.TRACKBAR_IMG, self.IMAGE_WINDOW_NAME, minimum_image_id)
            # print('The active_image_index after modification is %d '%( self.active_image_index, ))

        self.set_cluster_id_additional_operation()

    def set_cluster_id_additional_operation(self):
        pass

    def init_image_window_and_mouse_listener(self):
        self.init_window_size_font_size()
        # create window
        cv2.namedWindow(self.IMAGE_WINDOW_NAME, cv2.WINDOW_KEEPRATIO)  # cv2.WINDOW_KEEPRATIO cv2.WINDOW_AUTOSIZE
        cv2.resizeWindow(self.IMAGE_WINDOW_NAME, self.window_width, self.window_height)
        cv2.setMouseCallback(self.IMAGE_WINDOW_NAME, self.mouse_listener_for_image_window)

        # show the image index bar, self.set_image_index is defined in AnnotationTool()
        cv2.createTrackbar(self.TRACKBAR_IMG, self.IMAGE_WINDOW_NAME, 0, self.LAST_IMAGE_INDEX,
                           self.set_image_index)
        # if self.LAST_DIRECTORY_PAGE_INDEX != 0:
        #     cv2.createTrackbar(self.TRACKBAR_DIRECTORY, self.DIRECTORY_WINDOW_NAME,
        #                         0, self.LAST_DIRECTORY_PAGE_INDEX,
        #                         self.set_directory_page_index)

    def mouse_listener_for_image_window(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_x = x
            self.mouse_y = y
            # print('mouse move,  EVENT_MOUSEMOVE')

        # elif event == cv2.EVENT_LBUTTONDOWN:
        #     print('left click, EVENT_LBUTTONDOWN')

    # initialize the object colors
    def init_object_colors(self):
        self.object_colors = smrc.utils.CLASS_BGR_COLORS
        # print('CLASS_BGR_COLORS =', self.CLASS_BGR_COLORS)

        # If there are still more classes, add new colors randomly
        num_colors_missing = len(self.clusters) - len(self.object_colors)
        if num_colors_missing > 0:
            more_colors = np.random.randint(0, 255 + 1, size=(num_colors_missing, 3))
            self.object_colors = np.vstack([self.object_colors, more_colors])

    def estimate_display_object_id(self, class_list):
        """
        Transfer the object id (cluster id) to the object name (e.g., person 3, car 4) we display
        :param class_list:
        :return: dict, key global_bbox_id, value:[class_idx, class_object_count[class_idx]]
            e.g., person 3
        """

        self._assert_clusters_non_empty()

        # dict.fromkeys(self.video_detected_bbox_all.keys(), None)
        # no need to specify the keys for all detections
        self.object_id_to_display = {}
        # num_bbox = len(self.video_detected_bbox_all)
        # np.array([None] * num_bbox * 2).reshape((num_bbox, 2))

        print(f'Number of clusters are {len(self.clusters)}')
        self.cluster_labels = self.estimate_cluster_label(self.clusters)

        class_object_count = np.zeros((len(class_list),), dtype=np.int16)

        for cluster_id, cluster_ele in enumerate(self.clusters):
            # print('====================================================')
            # print('Transferring the bbox in cluster %d ' %(cluster_id,) )
            class_idx = self.cluster_labels[cluster_id]
            for global_bbox_id in cluster_ele:
                # bbox = self.get_single_bbox(global_bbox_id)
                # class_idx = bbox[0]

                # save the cluster label instead of the detected class label
                self.object_id_to_display[global_bbox_id] = [
                    class_idx, class_object_count[class_idx]
                ]
            class_object_count[class_idx] += 1
            # print(f'cluster_id = {cluster_id}, class_idx = {class_idx}, class_object_count = {class_object_count}')

    def draw_clustered_object_trajectory_online(self, tmp_img, cur_image_id, history_length=30):
        """
        Draw the trajectory of the object
        pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
        pts = pts.reshape((-1,1,2))
        cv.polylines(img,[pts],True,(0,255,255))
        :param tmp_img:
        :param cur_image_id:
        :param history_length:
        :return:
        """
        # if image_index_end is None:
        #     image_index_end = len(self.IMAGE_PATH_LIST)

        self._assert_clusters_non_empty()
        if len(self.object_colors) == 0:
            self.init_object_colors()

        if len(self.clusters_trajectory) == 0:
            self.init_clusters_trajectory()

        def pt2tuple(pt):
            return tuple(pt.reshape(2))

        for i, cluster in enumerate(self.clusters):
            # get valid bbox members
            image_ids = np.array(self.get_image_id_list_for_cluster(cluster))
            valid_ids = (image_ids <= cur_image_id) & (image_ids >= cur_image_id - history_length)
            # cluster_remains = list(np.array(cluster)[valid_ids])
            pointsInside = self.clusters_trajectory[i][valid_ids]
            # skip the plot if there are less than two detections
            if len(pointsInside) < 2: continue

            object_color = tuple([int(x) for x in self.object_colors[i]])
            # print(f'object_color = {object_color}')

            # cv2.polylines(tmp_img, [pts], True, (0, 255, 0), 3)  # tuple(object_color)
            for index, item in enumerate(pointsInside[:-1]):
                cv2.line(tmp_img, pt2tuple(item), pt2tuple(pointsInside[index + 1]), object_color, 2)

    # cv2.line(tmp_img, (0, y), (width, y), color, line_thickness)
    def init_clusters_trajectory(self):
        """Estimate the trajectories for all cluster.
        :return: shape (m, 1, 2), m is the number of detections
        """
        self.clusters_trajectory = [None] * len(self.clusters)
        for i, cluster in enumerate(self.clusters):
            xo_yo_pts = [smrc.utils.get_bbox_rect_center(self.get_single_bbox(global_bbox_id))
                         for global_bbox_id in cluster]
            pts = np.array(xo_yo_pts, np.int32)
            pts = pts.reshape((-1, 1, 2))
            self.clusters_trajectory[i] = pts

    def generate_video_for_tracking_result(
            self, blank_bg=False,
            fps=30, resulting_dir='tracking_results',
            duration=None, video_id_str=None
    ):
        self.post_process_after_offline_tracking()
        print('Start saving the object_tracking results as video ...')

        if video_id_str is None:
            image_name = self.IMAGE_PATH_LIST[0]
            image_dir_split = image_name.split(os.path.sep)
            image_dir = image_dir_split[0]
            video_id_str = '_'.join(image_dir_split[1:-1])

        dir_to_save = os.path.join(resulting_dir, video_id_str)
        smrc.utils.generate_dir_if_not_exist(dir_to_save)

        width = cv2.imread(self.IMAGE_PATH_LIST[0]).shape[1]
        if width > 800:
            self.LINE_THICKNESS = 2
            self.class_name_font_scale = 0.8
        else:
            self.LINE_THICKNESS = 1
            self.class_name_font_scale = 0.6

        pbar = tqdm(self.IMAGE_PATH_LIST)
        for image_path in pbar:
            if blank_bg:
                tmp = cv2.imread(image_path)
                h, w, _ = tmp.shape
                tmp_img = smrc.utils.generate_blank_image(h, w)
            else:
                tmp_img = cv2.imread(image_path)

            # # draw original detection results
            # ann_path = self.get_annotation_path(image_path)
            # self.draw_bboxes_from_file(tmp_img, ann_path)
            self.visualize_clustered_object_on_active_image(
                image_path, tmp_img
            )  #  additional_drawing_flag='IoU'

            # cv2.imshow(self.IMAGE_WINDOW_NAME, tmp_img)
            # 'images/284010/0000.jpg'
            # replace 'images' with 'tracking_results'

            # image_path_new = image_path.replace(self.IMAGE_DIR, resulting_dir, 1)
            image_path_new = os.path.join(dir_to_save, os.path.split(image_path)[-1])
            pbar.set_description(f'Saving object_tracking results to {image_path_new}')

            cv2.imwrite(image_path_new, tmp_img)

        if duration is not None:
            fps = smrc.utils.estimate_fps_based_on_duration(len(self.IMAGE_PATH_LIST), duration)
        pathIn, pathOut = dir_to_save, dir_to_save + '.avi'
        print(f'Generating object_tracking results to {pathOut}')
        smrc.utils.convert_frames_to_video(pathIn, pathOut, fps)


class VisualizationDeprecated(Visualization):
    def __init__(self):
        super().__init__()

    def draw_cluster_single_bbox_original(self, tmp_img, bbox, object_id, text_color=(0, 0, 0)):
        """
        draw the original detection results, car 1, truck 2, ...
        :param tmp_img: img
        :param bbox: original detected bbox
        :param object_id: object id to display
        :param text_color: color of the text
        :return:
        """
        bbox_rect = bbox[1:5]
        class_idx = bbox[0]
        self.draw_cluster_single_bbox_rect(tmp_img, bbox_rect, class_idx, object_id, text_color)

    def draw_detected_bbox_trajectory_online(
            self, tmp_img, image_index_str=0, image_index_end=None
    ):
        """
        Draw the trajectory of the object
        :param tmp_img:
        :param image_index_str:
        :param image_index_end:
        :return:
        """
        if image_index_end is None:
            image_index_end = len(self.IMAGE_PATH_LIST)

        for img_path in self.IMAGE_PATH_LIST[image_index_str: image_index_end]:
            # add key to the detected_bbox_list_dict
            for global_bbox_id in self.frame_dets[img_path]:
                bbox = self.get_single_bbox(global_bbox_id)
                class_idx, xmin, ymin, xmax, ymax = bbox

                # draw bbox
                class_color = self.CLASS_BGR_COLORS[class_idx].tolist()
                cv2.rectangle(tmp_img, (xmin, ymin), (xmax, ymax), class_color, self.LINE_THICKNESS)
                # location_to_draw (x,y)

    def draw_active_tracked_bbox(self, tmp_img, tracked_bbox_rect, text=None):
        xmin, ymin, xmax, ymax = tracked_bbox_rect
        cv2.rectangle(tmp_img, (xmin, ymin), (xmax, ymax), self.WHITE, self.LINE_THICKNESS)

        # draw resizing anchors
        # self.draw_bbox_anchors(tmp_img, xmin, ymin, xmax, ymax, self.ACTIVE_BBOX_COLOR, self.LINE_THICKNESS)

        if text is not None:
            # class_name = text
            # # text_shadow_color = class_color,
            text_color = (0, 0, 0)  # , i.e., black
            # int((ymin + ymax) / 2.0)
            self.draw_class_name(tmp_img, (xmin, ymax + 25),
                                 text, self.ACTIVE_BBOX_COLOR,
                                 text_color)
