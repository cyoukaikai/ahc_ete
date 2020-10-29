import os
import cv2

from .annotation_tool import AnnotationTool
from .select_directory import SelectDirectory
from .ann_utils import decrease_index, increase_index
# from ..det.json_det_process import get_json_dir_list
from ..base import get_dir_list_in_directory, non_blank_lines


class VisualizeLabel(AnnotationTool):
    def __init__(self, image_dir, label_dir,
                 class_list_file, auto_load_directory=None, user_name=None
                 ):
        # inherit the variables and functions from AnnotationTool
        AnnotationTool.__init__(
            self, image_dir, label_dir, class_list_file
        )
        self.user_name = user_name
        self.IMAGE_DIR = image_dir
        self.LABEL_DIR = label_dir
        self.auto_load_directory = auto_load_directory
        self.class_list_file = class_list_file

        self.active_directory = None
        self.IMAGE_WINDOW_NAME = 'VisualizeBBoxTool'
        self.TRACKBAR_IMG = 'Image'

        # self.LINE_THICKNESS = 1
        # if self.class_list_file is not None and os.:
        #     self.init_class_list_and_class_color()
        self.DIRECTORY_LIST = []
        self.directory_list_file = None

        self.show_visualization_setting()

    def show_visualization_setting(self):
        print('===================================== Information for visualization')
        print('self.IMAGE_DIR:', self.IMAGE_DIR)
        print('self.LABEL_DIR:', self.LABEL_DIR)
        print('self.auto_load_directory:', self.auto_load_directory)
        print('self.class_list_file:', self.class_list_file)
        print('self.user_name:', self.user_name)
        print('self.CLASS_LIST:', self.CLASS_LIST)
        print('=====================================')

        # print(f'self.LINE_THICKNESS = {self.LINE_THICKNESS}')
        # import sys
        # sys.exit(0)
    def init_image_window_and_mouse_listener(self):
        # reset the window size, line thickness, font scale if the width of the first image
        # in the image path list is greater than 1000.
        # Here we assume all the images in the directory has the same size, if this is not the
        # case, we may need to design more advanced setting (load all images, or sampling 5 images from
        # the image path list, and then use the maximum size of the image)
        self.init_window_size_font_size()

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
            # print('mouse move,  EVENT_MOUSEMOVE')

        # elif event == cv2.EVENT_LBUTTONDOWN:
        #     print('left click, EVENT_LBUTTONDOWN')

    def visualize_results(self):
        # initialize the self.IMAGE_PATH_LIST, self.LAST_IMAGE_INDEX
        # self.load_image_sequence(self.active_directory)
        self.load_active_directory()

        # this function must be after self.load_image_sequence(), otherwise, the trackBar
        # for image list can not be initialized (as number of image not known)
        self.init_image_window_and_mouse_listener()

        # load the first image in the IMAGE_PATH_LIST, 
        # initilize self.active_image_index and related information for the active image
        self.set_image_index(0)

        while True:
            # copy the current image
            tmp_img = self.active_image.copy()

            # get annotation paths
            image_path = self.IMAGE_PATH_LIST[self.active_image_index]  # image_path is not a global variable
            self.active_image_annotation_path = self.get_annotation_path(image_path)
            # print('annotation_path=', annotation_path)

            # display annotated bboxes
            self.draw_bboxes_from_file(tmp_img, self.active_image_annotation_path)  # , image_width, image_height

            self.set_active_bbox_idx_if_NONE()

            # set the active directory based on mouse cursor
            # self.set_active_bbox_idx_based_on_mouse_position()
            self.draw_active_bbox(tmp_img)

            cv2.imshow(self.IMAGE_WINDOW_NAME, tmp_img)

            ''' Key Listeners START '''
            pressed_key = self.read_pressed_key()

            # 255 Linux or Windows, cv2.waitKeyEx() & 0xFF , -1
            # Windows cv2.waitKeyEx() or Linux  cv2.waitKey(), 0
            # Windows cv2.waitKey()
            if pressed_key != 255 and pressed_key != 0 and pressed_key != -1:
                # print('pressed_key=', pressed_key)  # ('pressed_key=', -1) if no key is pressed.
                # print('pressed_key & 0xFF =', pressed_key & 0xFF)
                # print('self.platform = ', self.platform)
                # handle string key a -z
                if ord('a') <= pressed_key <= ord('z'):  # 'a': 97, 'z': 122
                    if pressed_key == ord('a') or pressed_key == ord('d'):
                        # show previous image key listener
                        if pressed_key == ord('a'):
                            self.active_image_index = decrease_index(
                                self.active_image_index, self.LAST_IMAGE_INDEX
                            )
                        # show next image key listener
                        elif pressed_key == ord('d'):
                            self.active_image_index = increase_index(
                                self.active_image_index, self.LAST_IMAGE_INDEX
                            )
                        cv2.setTrackbarPos(
                            self.TRACKBAR_IMG, self.IMAGE_WINDOW_NAME, self.active_image_index
                        )
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

    def init_directory_list(self):
        self.DIRECTORY_LIST = []
        # generate directory_list.txt automatically
        if self.auto_load_directory is not None and self.auto_load_directory in ['image_dir', 'label_dir']:
            if self.auto_load_directory == 'image_dir':
                self.DIRECTORY_LIST = get_dir_list_in_directory(
                    self.IMAGE_DIR
                )
            elif self.auto_load_directory == 'label_dir':
                self.DIRECTORY_LIST = get_dir_list_in_directory(
                    self.LABEL_DIR
                )
            # this will make the code too complex
            # elif self.auto_load_directory == 'json_dir':
            #     self.DIRECTORY_LIST = get_json_dir_list(
            #         self.JSON
            #     )
            else:
                print(f'self.auto_load_director should be in "image_dir", "label_dir"')
        else:
            if self.user_name is None or len(self.user_name) == 0:
                self.directory_list_file = 'directory_list.txt'
            else:
                self.user_name = self.user_name.lower()
                self.directory_list_file = 'directory_list_' + self.user_name + '.txt'

            # create the directory or file if they do not exist
            if not os.path.isfile(self.directory_list_file):
                open(self.directory_list_file, 'a').close()
            else:
                # load the directory list (check if them exists, if not ,
                # do not add them in the window, but show some information
                with open(self.directory_list_file) as f_directory_list:
                    directory_list = list(non_blank_lines(f_directory_list))
                # sorted(list(self.non_blank_lines(f_directory_list)), key=self.natural_sort_key)
                f_directory_list.close()  # close the file

                for ann_dir in directory_list:
                    # check if the directory exists
                    f_path = os.path.join(self.IMAGE_DIR, ann_dir)
                    assert os.path.isdir(f_path), \
                        'directory {}  does not exist, please check the file {}.'.format(
                            f_path, self.directory_list_file
                    )
                    self.DIRECTORY_LIST.append(ann_dir)
                print('DIRECTORY_LIST =', self.DIRECTORY_LIST)

    def main_loop(self):
        self.init_directory_list()

        while True:
            if self.active_directory is None:
                # a while loop until the self.active_directory is set
                # select directory to conduct object object_tracking
                # select_directory_tool = SelectDirectory(self.user_name, self.IMAGE_DIR, self.LABEL_DIR, 'label')
                # self.active_directory = select_directory_tool.set_active_directory()
                select_directory_tool = SelectDirectory(self.DIRECTORY_LIST)
                self.active_directory = select_directory_tool.set_active_directory()

            # annotate the active directory if it is set
            if self.active_directory is not None:
                print('Start visualize the annotation or object_detection results for '
                      'directory {}'.format(self.active_directory))

                print('self.CLASS_LIST = ', self.CLASS_LIST)
                # # the results are written in files (we can read them from disk)
                # tracker.object_tracking(self.active_directory, self.tracking_method)
                
                # visualize the object_tracking results
                self.visualize_results()

                # reinitialize the self.active_directory
                self.active_directory = None

        # # if quit the annotation tool, close all windows
        # cv2.destroyAllWindows()



