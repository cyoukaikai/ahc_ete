import os
import cv2
from tqdm import tqdm
# from .annotation_tool import AnnotationTool
from .select_directory import SelectDirectory
from .ann_utils import decrease_index, increase_index
from .keyboard import KeyboardCoding

import smrc.utils


class ImageSequenceViewer:
    def __init__(self, image_dir, dir_list=None):

        self.IMAGE_DIR = image_dir
        smrc.utils.assert_dir_exist(self.IMAGE_DIR)

        if dir_list is None:
            dir_list = smrc.utils.get_dir_list_in_directory(self.IMAGE_DIR)
        assert len(dir_list) > 0
        self.DIRECTORY_LIST = dir_list

        self.IMAGE_PATH_LIST = []
        self.active_directory = None

        self.IMAGE_WINDOW_NAME = 'VisualizationTool'
        self.TRACKBAR_IMG = 'Image'
        self.window_width = 1000
        self.window_height = 700

        self.active_image_index = 0
        self.active_image = None
        self.active_image_height = None
        self.active_image_width = None
        self.mouse_x = None
        self.mouse_y = None
        self.LAST_IMAGE_INDEX = None

        self.keyboard_encoder = KeyboardCoding()
        self.keyboard = self.keyboard_encoder.keyboard

        self.init_WithQT()
        self.show_visualization_setting()

    def visualization_annotation(self, dir_list, result_dir=None):
        if result_dir is None:
            result_dir = self.IMAGE_DIR + '_annotation_visualization'
        pbar = tqdm(enumerate(dir_list))
        for dir_idx, dir_name in pbar:
            pbar.set_description(f'Process {dir_name} [{dir_idx}/{len(dir_list)}] ...')
            self.active_directory = dir_name

            # initialize the self.IMAGE_PATH_LIST, self.LAST_IMAGE_INDEX
            # self.load_image_sequence(self.active_directory)
            self.load_image_sequence()

            # this function must be after self.load_image_sequence(), otherwise, the trackBar
            # for image list can not be initialized (as number of image not known)
            # self.init_image_window_and_mouse_listener()

            # load the first image in the IMAGE_PATH_LIST,

            smrc.utils.generate_dir_if_not_exist(
                os.path.join(result_dir, self.active_directory)
            )
            for k, image_path in enumerate(self.IMAGE_PATH_LIST):
                self.set_image_index(k)

                # image_path = self.IMAGE_PATH_LIST[self.active_image_index]  # image_path is not a global variable
                # copy the current image
                tmp_img = self.active_image.copy()
                self.draw_needed_on_active_image(tmp_img)

                # cv2.imshow(self.IMAGE_WINDOW_NAME, tmp_img)
                # cv2.imshow(self.IMAGE_WINDOW_NAME, tmp_img)
                new_image_file = image_path.replace(self.IMAGE_DIR, result_dir)
                cv2.imwrite(new_image_file, tmp_img)

    def show_visualization_setting(self):
        print('===================================== Information for visualization')
        print('self.IMAGE_DIR:', self.IMAGE_DIR)
        print('self.DIRECTORY_LIST:', self.DIRECTORY_LIST)
        print('=====================================')

    def init_window_size(self):
        if len(self.IMAGE_PATH_LIST) > 0:
            image_path = self.IMAGE_PATH_LIST[0]
            tmp_image = cv2.imread(image_path)
            assert tmp_image is not None

            height, width, _ = tmp_image.shape
            # print(f'image size: height={height}, width={width} for {os.path.dirname(image_path)}')  # (400, 640, 3)
            self.window_width = width + 50   # 1000
            self.window_height = height + 50  # 700
            # change the setting for the window and line thickness if the image width > 1000
            # if width > 1000:
            #     self.window_width = 1300  # 1000
            #     self.window_height = 800  # 700
            # else:
            #     self.window_width = 1000
            #     self.window_height = 700

    def init_image_window_and_mouse_listener(self):
        # reset the window size, line thickness, font scale if the width of the first image
        # in the image path list is greater than 1000.
        # Here we assume all the images in the directory has the same size, if this is not the
        # case, we may need to design more advanced setting (load all images, or sampling 5 images from
        # the image path list, and then use the maximum size of the image)
        self.init_window_size()

        # create window
        cv2.namedWindow(self.IMAGE_WINDOW_NAME, cv2.WINDOW_KEEPRATIO)  # cv2.WINDOW_KEEPRATIO cv2.WINDOW_AUTOSIZE
        cv2.resizeWindow(self.IMAGE_WINDOW_NAME, self.window_width, self.window_height)
        cv2.setMouseCallback(self.IMAGE_WINDOW_NAME, self.mouse_listener_for_image_window)

        # show the image index bar, self.set_image_index is defined in AnnotationTool()
        if len(self.IMAGE_PATH_LIST) > 0:
            cv2.createTrackbar(
                self.TRACKBAR_IMG, self.IMAGE_WINDOW_NAME,
                0, self.LAST_IMAGE_INDEX,
                self.set_image_index
            )

    def set_image_index(self, ind):
        self.active_image_index = ind
        img_path = self.IMAGE_PATH_LIST[self.active_image_index]  # local variable
        # print(f'self.IMAGE_PATH_LIST = {self.IMAGE_PATH_LIST}')
        # print(f'img_path = {img_path}')
        self.active_image = cv2.imread(img_path)
        self.active_image_height, self.active_image_width = self.active_image.shape[:2]
        text = 'Showing image {}/{}, path: {}'.format(self.active_image_index, self.LAST_IMAGE_INDEX,
                                                      img_path)
        # self.display_text(text, 1000)
        # print(f'{text}')

    # directory_window_mouse_listener
    def mouse_listener_for_image_window(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_x = x
            self.mouse_y = y
            # print('mouse move,  EVENT_MOUSEMOVE')

        # elif event == cv2.EVENT_LBUTTONDOWN:
        #     print('left click, EVENT_LBUTTONDOWN')

    def load_image_sequence(self):
        image_dir = os.path.join(self.IMAGE_DIR, self.active_directory)

        self.IMAGE_PATH_LIST = []

        # load image list
        for f in sorted(os.listdir(image_dir), key=smrc.utils.natural_sort_key):
            f_path = os.path.join(image_dir, f)
            if os.path.isdir(f_path):
                # skip directories
                continue

            # check if it is an image
            # Windows will generate non-image files in the image directory, e.g.,
            # img_path = Taxi_SampleData/638000/Thumbs.db
            # we have to check in advance, otherwise, the number of images are wrong
            # and when we move the last image, non image file will be tried to load
            # and cause NoneType error for later image annotation
            if smrc.utils.is_image(f_path):
                self.IMAGE_PATH_LIST.append(f_path)
        self.LAST_IMAGE_INDEX = len(self.IMAGE_PATH_LIST) - 1

    def view_active_directory(self):
        # initialize the self.IMAGE_PATH_LIST, self.LAST_IMAGE_INDEX
        # self.load_image_sequence(self.active_directory)
        self.load_image_sequence()

        # this function must be after self.load_image_sequence(), otherwise, the trackBar
        # for image list can not be initialized (as number of image not known)
        self.init_image_window_and_mouse_listener()

        # load the first image in the IMAGE_PATH_LIST,
        # initilize self.active_image_index and related information for the active image
        self.set_image_index(0)

        while True:
            # copy the current image
            tmp_img = self.active_image.copy()

            self.draw_needed_on_active_image(tmp_img)

            pressed_key = self.read_pressed_key()
            if pressed_key & 0xFF == 27:  # Esc key is pressed
                # cv2.destroyWindow(self.IMAGE_WINDOW_NAME)
                break
            else:
                self.keyboard_listener(pressed_key, tmp_img)

            cv2.imshow(self.IMAGE_WINDOW_NAME, tmp_img)

            if self.WITH_QT:
                # if window gets closed then quit
                if cv2.getWindowProperty(self.IMAGE_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    # close the window
                    cv2.destroyWindow(self.IMAGE_WINDOW_NAME)
                    break

    def init_WithQT(self):
        try:
            cv2.namedWindow('Test')
            cv2.displayOverlay('Test', 'Test QT', 500)
            self.WITH_QT = True
        except cv2.error:
            print('-> Please ignore this error message\n')
        cv2.destroyAllWindows()

    def keyboard_listener(self, pressed_key, tmp_img=None):
        ''' Key Listeners START '''

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

    def draw_needed_on_active_image(self, tmp_img):
        pass

    def read_pressed_key(self):
        return self.keyboard_encoder.read_pressed_key()

    def main_loop(self):
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
                print('Start visualize directory {}'.format(self.active_directory))

                # visualize the object_tracking results
                self.view_active_directory()

                # reinitialize the self.active_directory
                self.active_directory = None


