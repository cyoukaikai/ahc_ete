import os
import cv2
import sys

from .annotation_tool import AnnotationTool
import smrc.utils


class VisualizeOutlier(AnnotationTool):
    def __init__(
            self, image_dir='images',
            label_dir='labels',
            class_list_file='class_list.txt',
            specified_image_list=None,
            specified_ann_path_list=None
    ):
        AnnotationTool.__init__(self)
        self.IMAGE_DIR= image_dir
        self.LABEL_DIR = label_dir
        self.class_list_file = class_list_file

        # self.active_directory = None
        self.specified_image_list = specified_image_list
        self.specified_ann_path_list = specified_ann_path_list

        self.IMAGE_WINDOW_NAME = 'VisualizeSpecifiedData'
        self.TRACKBAR_IMG = 'Image'

        self.active_image_index = None
        # print('self.IMAGE_DIR :',self.IMAGE_DIR)
        # print('self.LABEL_DIR :',self.LABEL_DIR)

        if self.class_list_file is not None and os.path.isfile(self.class_list_file):
            self.init_class_list_and_class_color()
        #     # print(self.CLASS_LIST)
        #     # sys.exit(0)

    def init_image_window_and_mouse_listener(self):
        # create window
        cv2.namedWindow(self.IMAGE_WINDOW_NAME, cv2.WINDOW_KEEPRATIO)  # cv2.WINDOW_KEEPRATIO cv2.WINDOW_AUTOSIZE
        cv2.resizeWindow(self.IMAGE_WINDOW_NAME, self.window_width, self.window_height)
        cv2.setMouseCallback(self.IMAGE_WINDOW_NAME, self.mouse_listener_for_image_window)
    
        # show the image index bar, self.set_image_index is defined in AnnotationTool()
        cv2.createTrackbar(self.TRACKBAR_IMG, self.IMAGE_WINDOW_NAME, 0, self.LAST_IMAGE_INDEX, self.set_image_index)

    # directory_window_mouse_listener
    def mouse_listener_for_image_window(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_x = x
            self.mouse_y = y
            #print('mouse move,  EVENT_MOUSEMOVE')

        # elif event == cv2.EVENT_LBUTTONDOWN:
        #     print('left click, EVENT_LBUTTONDOWN')

    def load_specified_annotation_file_list(self):

        if self.specified_ann_path_list is None:
            print('self.specified_ann_path_list is None.')
            sys.exit(0)

        # print(self.specified_ann_path_list)
        if len(self.specified_ann_path_list) > 10000:
            self.specified_ann_path_list = self.specified_ann_path_list[:10000]

        self.IMAGE_PATH_LIST = []
        for f in self.specified_ann_path_list:
            # print(f'self.LABEL_DIR = {self.LABEL_DIR}.')
            # print(f'self.IMAGE_DIR = {self.IMAGE_DIR}.')
            f_path = smrc.utils.get_image_or_annotation_path(
                f, self.LABEL_DIR, self.IMAGE_DIR, '.jpg'
            )
            # print(f'image_name  = {f_path}.')
            # check if it is an image
            test_img = cv2.imread(f_path)
            if test_img is not None:
                self.IMAGE_PATH_LIST.append(f_path)
            else:
                print(f'load_specified_annotation_file_list: {f_path} does not exist.')
                sys.exit(0)
        self.LAST_IMAGE_INDEX = len(self.IMAGE_PATH_LIST) - 1
        print('{} images are loaded to self.IMAGE_PATH_LIST'.format(len(self.IMAGE_PATH_LIST)))

    def load_specified_image_list(self):
        if self.specified_image_list is None:
            print('self.specified_image_list is None.')
            sys.exit(0)

        if len(self.specified_image_list) > 1000:
            self.specified_image_list = self.specified_image_list[:5000]

        self.IMAGE_PATH_LIST = []
        
        for f_path in self.specified_image_list:
            # check if it is an image
            test_img = cv2.imread(f_path)
            if test_img is not None:
                self.IMAGE_PATH_LIST.append(f_path)
            else:
                print(f'{f_path} does not exist.')
                sys.exit(0)
            ann_path = smrc.utils.get_image_or_annotation_path(
                f_path, self.IMAGE_DIR, self.LABEL_DIR, '.txt'
            )
            if not os.path.isfile(ann_path):
                print(f'{ann_path} does not exist.')
        #         open(ann_path, 'a').close()
        self.LAST_IMAGE_INDEX = len(self.IMAGE_PATH_LIST) - 1
        print('{} images are loaded to self.IMAGE_PATH_LIST'.format(len(self.IMAGE_PATH_LIST)))

    # annotation_or_detection
    def visualize_specified_results(self):

        # this function must be after self.load_image_sequence(), otherwise, the trackBar
        # for image list can not be initialized (as number of image not known)
        self.init_image_window_and_mouse_listener()

        # load the first image in the IMAGE_PATH_LIST, 
        # initilize self.active_image_index and related information for the active image
        self.set_image_index(0)

        while True:
            # load the class index and class color for plot
            color = self.CLASS_BGR_COLORS[self.active_class_index].tolist()

            # copy the current image
            tmp_img = self.active_image.copy()

            # get annotation paths
            image_path = self.IMAGE_PATH_LIST[self.active_image_index]  # image_path is not a global variable
            self.active_image_annotation_path = self.get_annotation_path(image_path)

            print('image_path=', image_path)
            print('annotation_path =', self.active_image_annotation_path)

            # display annotated bboxes
            self.draw_bboxes_from_file(tmp_img, self.active_image_annotation_path)  # , image_width, image_height
            self.set_active_bbox_idx_if_NONE()
            # set the active directory based on mouse cursor
            # self.set_active_directory_based_on_mouse_position()
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

    def main_loop(self):
        print('self.CLASS_LIST = ', self.CLASS_LIST)

        # load the image list from specified
        if self.specified_image_list is not None:
            self.load_specified_image_list()
        elif self.specified_ann_path_list is not None:
            self.load_specified_annotation_file_list()
        
        self.visualize_specified_results()

        # if quit the annotation tool, close all windows
        cv2.destroyAllWindows()




