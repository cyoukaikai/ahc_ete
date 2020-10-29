# import os
# import cv2
# import argparse
#
# import smrc.line
# import smrc.line.det.detection_process
# from .visualize_label import VisualizeLabel
# from smrc.line.det.json_det_process import load_json_detection_to_dict
# from smrc.line.det.detection_process import non_max_suppression_single_image, ScorePosition, get_det_index
#
# # not finished
# class VisualizeJsonDetection(VisualizeLabel):
#     def __init__(
#             self, image_dir,
#             class_list_file, json_file_dir
#             ):
#         # inherit the variables and functions from AnnotationTool
#         VisualizeLabel.__init__(
#             self, image_dir=image_dir,
#             label_dir=None,
#             class_list_file=class_list_file,
#             auto_load_directory=None,  # 'json_dir'
#             user_name=None
#         )
#         self.active_directory_detection_dict = {}
#         self.json_file_dir = json_file_dir
#         self.score_thd = 10  # 0.10
#         self.TRACKBAR_SCORE = 'Confidence Level'
#         self.show_score_on = True
#         self.nms_thd_on = True
#         self.nms_thd = 0.01
#         self.auto_adjust_score_thd_on = True
#
#     def init_image_window_and_mouse_listener(self):
#         # create window
#         cv2.namedWindow(self.IMAGE_WINDOW_NAME, cv2.WINDOW_KEEPRATIO)  # cv2.WINDOW_KEEPRATIO cv2.WINDOW_AUTOSIZE
#         cv2.resizeWindow(self.IMAGE_WINDOW_NAME, self.window_width, self.window_height)
#         cv2.setMouseCallback(self.IMAGE_WINDOW_NAME, self.mouse_listener_for_image_window)
#
#         # show the image index bar, self.set_image_index is defined in AnnotationTool()
#         cv2.createTrackbar(self.TRACKBAR_IMG, self.IMAGE_WINDOW_NAME, 0, self.LAST_IMAGE_INDEX, self.set_image_index)
#         cv2.createTrackbar(self.TRACKBAR_SCORE, self.IMAGE_WINDOW_NAME, 0, 100, self.set_score_thd)
#
#     def load_active_directory_additional_operation(self):
#         json_detection_file = os.path.join(
#             self.json_file_dir,
#             self.active_directory + '.json'  # 'smrc_' +
#         )
#         assert os.path.isfile(json_detection_file), \
#             'json_detection_file {json_detection_file} does not exist ...'
#
#         self.active_directory_detection_dict = load_json_detection_to_dict(
#             json_detection_file,
#             detection_dict=None,
#             score_thd=0.0,
#             short_image_path=True
#         )
#
#     def set_score_thd(self, ind):
#         self.score_thd = ind
#         cv2.setTrackbarPos(self.TRACKBAR_SCORE, self.IMAGE_WINDOW_NAME,
#                            self.score_thd)
#
#     def draw_bboxes_from_file(self, tmp_img, score_thd):
#         '''
#         # load the draw bbox from file, and initialize the annotated bbox in this image
#         for fast accessing (annotation -> self.active_image_annotated_bboxes)
#             ann_path = labels/489402/0000.txt
#             print('this ann_path =', ann_path)
#             detection_dict[image_path] = [ [class_idx, xmin, ymin, xmax, ymax, score], ... ]
#         '''
#         image_path_full = self.IMAGE_PATH_LIST[self.active_image_index]
#         self.active_image_annotated_bboxes = []  # initialize the image_annotated_bboxes
#
#         # print(f'self.active_directory_detection_dict = {self.active_directory_detection_dict}')
#
#         image_dir, video_dir, last_name = smrc.line.split_image_path(image_path_full)
#         image_path = os.path.join(video_dir, last_name)
#         # print(f'image_path = {image_path}')
#
#         if image_path in self.active_directory_detection_dict:
#             detection_list = self.active_directory_detection_dict[image_path]
#             if self.nms_thd_on and 0 <= self.nms_thd <= 1:
#                 detection_list = non_max_suppression_single_image(
#                     image_pred=detection_list, nms_thd=self.nms_thd, score_position=ScorePosition.Last)
#
#             for detection in detection_list:
#                 bbox, score = list(map(int, detection[:5])), detection[
#                     get_det_index(score_position=ScorePosition.Last)
#                 ]
#                 if score > score_thd:
#                     self.active_image_annotated_bboxes.append(bbox)
#                     if self.show_score_on:
#                         self.draw_annotated_bbox(tmp_img, bbox, score)
#                     else:
#                         self.draw_annotated_bbox(tmp_img, bbox)
#
#     def draw_annotated_bbox(self, tmp_img, bbox,  score=None):
#
#         # the data format should be int type, class_idx is 0-index.
#         class_idx, xmin, ymin, xmax, ymax = bbox
#         # draw bbox
#         class_color = self.CLASS_BGR_COLORS[class_idx].tolist()
#         self.draw_single_bbox(tmp_img, bbox, class_color, class_color, class_color)
#         # draw_single_bbox(self, tmp_img, bbox, rectangle_color, anchor_rect_color, text_shadow_color)
#
#         text_shadow_color = class_color
#         text_color = (0, 0, 0)
#         cv2.rectangle(tmp_img, (xmin, ymin), (xmax, ymax), class_color, self.LINE_THICKNESS)
#         # draw resizing anchors
#         self.draw_bbox_anchors(tmp_img, xmin, ymin, xmax, ymax, class_color, self.LINE_THICKNESS)
#
#         class_name = self.CLASS_LIST[class_idx]
#         if score is not None:
#             # "%.2f" % a
#             class_name = class_name + ' %.2f' % score
#         # text_shadow_color = class_color, text_color = (0, 0, 0), i.e., black
#         self.draw_class_name(tmp_img, (xmin, ymin), class_name, text_shadow_color, text_color)  #
#
#     def set_score_thd_based_on_image_size(self):
#         img_height, img_width = self.active_image_height, self.active_image_width
#
#         if img_width > 1000:
#             self.set_score_thd(5)
#         else:
#             self.set_score_thd(20)
#
#     def view_active_directory(self):
#         # initialize the self.IMAGE_PATH_LIST, self.LAST_IMAGE_INDEX
#         self.load_image_sequence()
#
#         # this function must be after self.load_image_sequence(), otherwise, the trackBar
#         # for image list can not be initialized (as number of image not known)
#         self.init_image_window_and_mouse_listener()
#
#         # load the first image in the IMAGE_PATH_LIST,
#         # initilize self.active_image_index and related information for the active image
#         self.set_image_index(0)
#         if self.auto_adjust_score_thd_on:
#             self.set_score_thd_based_on_image_size()
#         else:
#             self.set_score_thd(self.score_thd)
#
#         while True:
#             # load the class index and class color for plot
#             color = self.CLASS_BGR_COLORS[self.active_class_index].tolist()
#
#             # copy the current image
#             tmp_img = self.active_image.copy()
#
#             img_path = self.IMAGE_PATH_LIST[self.active_image_index]
#             image_name = img_path.split(os.path.sep)[-1]
#             text_content = f'{self.active_directory}/{image_name} \n'
#                 # f'Press v to show or not show score'
#             smrc.line.display_text_on_image(tmp_img, text_content)
#             # display annotated bboxes
#             self.draw_bboxes_from_file(tmp_img, self.score_thd / 100.0)
#             self.set_active_bbox_idx_if_NONE()
#
#             # set the active directory based on mouse cursor
#             self.set_active_bbox_idx_based_on_mouse_position()
#             self.draw_active_bbox(tmp_img)
#
#             cv2.imshow(self.IMAGE_WINDOW_NAME, tmp_img)
#
#             ''' Key Listeners START '''
#             pressed_key = self.read_pressed_key()
#
#             # 255 Linux or Windows, cv2.waitKeyEx() & 0xFF , -1  Windows
#             # cv2.waitKeyEx() or Linux  cv2.waitKey(), 0 Windows cv2.waitKey()
#             if pressed_key != 255 and pressed_key != 0 and pressed_key != -1:
#                 # print('pressed_key=', pressed_key)  # ('pressed_key=', -1) if no key is pressed.
#                 # print('pressed_key & 0xFF =', pressed_key & 0xFF)
#                 # print('self.platform = ', self.platform)
#                 # handle string key a -z
#                 if ord('a') <= pressed_key <= ord('z'):  # 'a': 97, 'z': 122
#                     if pressed_key == ord('a') or pressed_key == ord('d'):
#                         # show previous image key listener
#                         if pressed_key == ord('a'):
#                             self.active_image_index = smrc.line.decrease_index(self.active_image_index,
#                                                                                 self.LAST_IMAGE_INDEX)
#                         # show next image key listener
#                         elif pressed_key == ord('d'):
#                             self.active_image_index = smrc.line.increase_index(self.active_image_index,
#                                                                                 self.LAST_IMAGE_INDEX)
#                         cv2.setTrackbarPos(self.TRACKBAR_IMG, self.IMAGE_WINDOW_NAME, self.active_image_index)
#                     elif pressed_key == ord('j') or pressed_key == ord('k'):
#                         # show previous image key listener
#                         if pressed_key == ord('j'):
#                             self.score_thd = smrc.line.decrease_index(
#                                 self.score_thd,100)
#                         # show next image key listener
#                         elif pressed_key == ord('k'):
#                             self.score_thd = smrc.line.increase_index(
#                                 self.score_thd, 100
#                                 )
#                         cv2.setTrackbarPos(self.TRACKBAR_SCORE, self.IMAGE_WINDOW_NAME, self.score_thd)
#
#                     elif pressed_key == ord('v'):
#                         self.show_score_on = not self.show_score_on
#
#                 elif pressed_key & 0xFF == 27:  # Esc key is pressed
#                     # close the window
#                     cv2.destroyWindow(self.IMAGE_WINDOW_NAME)
#                     break
#
#             if self.WITH_QT:
#                 # if window gets closed then quit
#                 if cv2.getWindowProperty(self.IMAGE_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
#                     # close the window
#                     cv2.destroyWindow(self.IMAGE_WINDOW_NAME)
#                     break
