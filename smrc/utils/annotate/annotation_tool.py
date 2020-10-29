import os
import cv2
import numpy as np
import sys

import smrc.utils
from .keyboard import KeyboardCoding


class Color:
    # import smrc.line.color
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    ACTIVE_BBOX_COLOR = (0, 255, 255)  # yellow, the RGB of yellow is (255, 255,0 ), but here we use BGR format
    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    YELLOW = (0, 255, 255)

    @staticmethod
    def init_class_color(class_list):
        """Define the colors for the classes
        The colors are in BGR order because we're using OpenCV.
        :param class_list:
        :return:
        """
        class_colors = smrc.utils.CLASS_BGR_COLORS

        # If there are still more classes, add new colors randomly
        num_colors_missing = len(class_list) - len(class_colors)
        if num_colors_missing > 0:
            more_colors = np.random.randint(0, 255 + 1, size=(num_colors_missing, 3))
            class_colors = np.vstack([class_colors, more_colors])
        return class_colors


class ActiveImage:
    def __init__(self):
        self.active_image = None  # the pixel values of an image, this will be the image we are operating
        self.active_image_height = None  # height of img
        self.active_image_width = None  # width of img

        # the annotation path of the active image, this variable will be used through the operation of the active image
        self.active_image_annotation_path = None
        self.active_image_annotated_bboxes = []  # a list the bboxes annotated in this image
        self.active_bbox_idx = None  # the bbox id of the selected bbox in image_annotated_bboxes

    def is_valid_active_bbox(self):
        return self.active_bbox_idx is not None and \
               self.active_image_annotated_bboxes is not None and \
               self.active_bbox_idx < len(self.active_image_annotated_bboxes)

    def get_active_bbox(self):
        if self.is_valid_active_bbox():
            return self.active_image_annotated_bboxes[self.active_bbox_idx]
        else:
            return None


class BboxAnchor:
    def __init__(self):
        self.ANCHOR_POSITION_ENCODING = [['L', 'O', 'C', 'O', 'R'],  # horizontal: left, other, center, other, right
                                         ['T', 'O', 'M', 'O', 'B']]  # vertical: top, other, middle, other, bottom
        self.ACTIVE_ANCHOR_DICTIONARY = {}
        self.active_anchor_position = None
        self.active_anchor_arrow_length = 10  # the length of the arrow in the horizontal and vertical direction
        self.active_anchor_arrow_magnitude = 5

        # Size of resizing anchors (depends on LINE_THICKNESS)
        self.BBOX_ANCHOR_THICKNESS = 4  # self.LINE_THICKNESS * 2 # it is used by get_active_anchor region

        # initialize the anchor dictionary (definition of 8 anchor positions)
        # for later annotation use
        self.init_active_anchor_dictionary()

    def init_active_anchor_dictionary(self):
        # -> Left, Other, ..., Center, Other, ..., Right
        # LT OT OT OT CT OT OT OT RT  #Top
        # LO                      RO
        # LO                      RO
        # LO                      RO
        # LO                      RO
        # LM                      RM  #Middle
        # LO                      RO
        # LO                      RO
        # LO                      RO
        # LO                      RO
        # LO                      RO
        # LB OB OB OB CB OB OB OB RB # Bottom

        # the length of the arrow in the horizontal and vertical direction
        arrow_length_horizontal_vertical = self.active_anchor_arrow_length

        # the length of the arrow in the diagonal direction
        arrow_length_diagonal = int(arrow_length_horizontal_vertical * np.sqrt(2) / 2.0)
        shift_east, shift_west = (-arrow_length_horizontal_vertical, 0), (arrow_length_horizontal_vertical, 0)
        shift_north, shift_south = (0, -arrow_length_horizontal_vertical), (0, arrow_length_horizontal_vertical)

        shift_northeast = [(arrow_length_diagonal, -arrow_length_diagonal),
                           (-arrow_length_diagonal, arrow_length_diagonal)]
        shift_southwest = shift_northeast

        shift_northwest = [(-arrow_length_diagonal, -arrow_length_diagonal),
                           (arrow_length_diagonal, arrow_length_diagonal)]
        shift_southeast = shift_northwest
        # diagonal direction
        self.ACTIVE_ANCHOR_DICTIONARY['LT'] = shift_northwest
        self.ACTIVE_ANCHOR_DICTIONARY['RT'] = shift_northeast
        self.ACTIVE_ANCHOR_DICTIONARY['LB'] = shift_southwest
        self.ACTIVE_ANCHOR_DICTIONARY['RB'] = shift_southeast

        # move the center of the bbox
        self.ACTIVE_ANCHOR_DICTIONARY['LO'] = [shift_east, shift_west, shift_north, shift_south]
        self.ACTIVE_ANCHOR_DICTIONARY['RO'] = [shift_east, shift_west, shift_north, shift_south]
        self.ACTIVE_ANCHOR_DICTIONARY['OT'] = [shift_east, shift_west, shift_north, shift_south]
        self.ACTIVE_ANCHOR_DICTIONARY['OB'] = [shift_east, shift_west, shift_north, shift_south]

        # move one of four segments of the bbox
        self.ACTIVE_ANCHOR_DICTIONARY['LM'] = [shift_east, shift_west]  # left boundary
        self.ACTIVE_ANCHOR_DICTIONARY['RM'] = [shift_east, shift_west]  # right boundary

        self.ACTIVE_ANCHOR_DICTIONARY['CT'] = [shift_north, shift_south]  # top boundary
        self.ACTIVE_ANCHOR_DICTIONARY['CB'] = [shift_north, shift_south]  # bottom boundary

        # there are four illegal positions
        self.ACTIVE_ANCHOR_DICTIONARY['CM'] = None
        self.ACTIVE_ANCHOR_DICTIONARY['OO'] = None
        self.ACTIVE_ANCHOR_DICTIONARY['OM'] = None
        self.ACTIVE_ANCHOR_DICTIONARY['CO'] = None
        # print(self.ACTIVE_ANCHOR_DICTIONARY)

    def get_bbox_active_region_rectangle(self, xmin, ymin, xmax, ymax): #, bbox_anchor_thickness
        # xmin = xmin - self.BBOX_ANCHOR_THICKNESS
        # ymin = ymin - self.BBOX_ANCHOR_THICKNESS
        # xmax = xmax + self.BBOX_ANCHOR_THICKNESS
        # ymax = ymax + self.BBOX_ANCHOR_THICKNESS

        xmin = xmin - self.BBOX_ANCHOR_THICKNESS
        ymin = ymin - self.BBOX_ANCHOR_THICKNESS
        xmax = xmax + self.BBOX_ANCHOR_THICKNESS
        ymax = ymax + self.BBOX_ANCHOR_THICKNESS
        return [xmin, ymin, xmax, ymax]


class Draw:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.LINE_THICKNESS = 1  # the thickness of the line of the bbox
        self.class_name_font_scale = 0.6

        self.BBOX_WIDTH_OR_HEIGHT_THRESHOLD = 10  # to decide the smallest bbox width or height we allowed to draw.
        # we use line thickness of self.LINE_THICKNESS * 2 for a larger bbox
        self.bbox_area_threshold_for_thick_line = 10000

        ###############################################
        # not for actual use
        ################################################
        self.CLASS_LIST = None
        self.CLASS_BGR_COLORS = None

    @staticmethod
    def draw_edges(tmp_img):
        blur = cv2.bilateralFilter(tmp_img, 3, 75, 75)
        edges = cv2.Canny(blur, 150, 250, 3)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        # Overlap image and edges together
        tmp_img = np.bitwise_or(tmp_img, edges)
        # tmp_img = cv2.addWeighted(tmp_img, 1 - edges_val, edges, edges_val, 0)
        return tmp_img  # have to return tmp_img

    def draw_single_bbox(self, tmp_img, bbox, rectangle_color, anchor_rect_color,
                         text_shadow_color, text_color=(0, 0, 0)):
        # the data format should be int type, class_idx is 0-index.
        class_idx, xmin, ymin, xmax, ymax = bbox
        if smrc.utils.get_bbox_area(xmin, ymin, xmax, ymax) < self.bbox_area_threshold_for_thick_line:
            cv2.rectangle(tmp_img, (xmin, ymin), (xmax, ymax), rectangle_color, self.LINE_THICKNESS)
            # draw resizing anchors
            self.draw_bbox_anchors(tmp_img, xmin, ymin, xmax, ymax, anchor_rect_color, self.LINE_THICKNESS)
        else:
            cv2.rectangle(tmp_img, (xmin, ymin), (xmax, ymax), rectangle_color, 2)
            self.draw_bbox_anchors(tmp_img, xmin, ymin, xmax, ymax, anchor_rect_color, 2)

        # print(f'self.bbox_area_threshold_for_thick_line = {self.bbox_area_threshold_for_thick_line}')
        # sys.exit(0)
        class_name = self.CLASS_LIST[class_idx]
        # text_shadow_color = class_color, text_color = (0, 0, 0), i.e., black
        self.draw_class_name(tmp_img, (xmin, ymin), class_name, text_shadow_color, text_color)  #

    # draw the 8 bboxes of the anchors around the bbox
    def draw_bbox_anchors(self, tmp_img, xmin, ymin, xmax, ymax, anchor_color, line_thickness=None):
        if line_thickness is None:
            line_thickness = self.LINE_THICKNESS

        anchor_dict = smrc.utils.get_anchors_rectangles(xmin, ymin, xmax, ymax, line_thickness)
        for anchor_key in anchor_dict:
            x1, y1, x2, y2 = anchor_dict[anchor_key]
            cv2.rectangle(tmp_img, (int(x1), int(y1)), (int(x2), int(y2)), anchor_color, -1)

    def draw_class_name(self, tmp_img, location_to_draw, text_content, text_shadow_color, text_color):

        font = cv2.FONT_HERSHEY_SIMPLEX  # FONT_HERSHEY_SIMPLEX
        font_scale = self.class_name_font_scale
        margin = 3
        text_width, text_height = cv2.getTextSize(text_content, font, font_scale, self.LINE_THICKNESS)[0]

        xmin, ymin = int(location_to_draw[0]), int(location_to_draw[1])
        cv2.rectangle(tmp_img, (xmin, ymin), (xmin + text_width + margin, ymin - text_height - margin),
                      text_shadow_color, -1)

        cv2.putText(tmp_img, text_content, (xmin, ymin - 5), font, font_scale, text_color, self.LINE_THICKNESS,
                    int(cv2.LINE_AA))

    # draw the normal annotated bbox on the active image
    # the active bbox will be redrawn by another function (with only the colors are changed)
    def draw_annotated_bbox(self, tmp_img, bbox):
        # the data format should be int type, class_idx is 0-index.
        class_idx = bbox[0]
        # draw bbox
        class_color = self.CLASS_BGR_COLORS[class_idx].tolist()
        self.draw_single_bbox(tmp_img, bbox, class_color, class_color, class_color)
        # draw_single_bbox(self, tmp_img, bbox, rectangle_color, anchor_rect_color, text_shadow_color)


class MainWindow(Color, ActiveImage, Draw, BboxAnchor):
    def __init__(self):
        # super(Color).__init__()
        # super(ActiveImage).__init__()
        # super(Draw).__init__()
        # super(BboxAnchor).__init__()
        Color.__init__(self)
        ActiveImage.__init__(self)
        Draw.__init__(self)
        BboxAnchor.__init__(self)

        self.IMAGE_WINDOW_NAME = ''  #
        # for plotting
        self.window_width = 1000
        self.window_height = 700
        self.WITH_QT = False

        self.mouse_x, self.mouse_y = 0, 0
        self.point_1, self.point_2 = None, None  # the first and the second point for a bbox

        ##########################################
        # only for definition, not for actual use
        ##########################################
        self.IMAGE_PATH_LIST = []
        self.class_list_file = None
        self.LAST_IMAGE_INDEX = None  # LAST_IMAGE_INDEX = len(IMAGE_PATH_LIST) - 1

        self.CLASS_LIST = []
        self.LAST_CLASS_INDEX = None  # no need this #LAST_CLASS_INDEX = len(CLASS_LIST) - 1
        self.CLASS_BGR_COLORS = []

        # record the last active bbox in the last operating image, not necessarily the previous image (could be
        # next image if we are moving backward)
        self.active_bbox_previous_image = None

        ###################################
        # do we set the values here or not ?
        self.active_class_index = 0
        self.active_class_index = None
        self.active_image_index = None
        self.last_added_bbox = None
        ###################################

        # with QT or not
        self.init_WithQT()

    def init_WithQT(self):
        try:
            cv2.namedWindow('Test')
            cv2.displayOverlay('Test', 'Test QT', 500)
            self.WITH_QT = True
        except cv2.error:
            print('-> Please ignore this error message\n')
        cv2.destroyAllWindows()

    # display text information on the top of the active image for a given period
    def display_text(self, text, display_period):
        if self.WITH_QT:
            cv2.displayOverlay(self.IMAGE_WINDOW_NAME, text, display_period)
        else:
            print(text)

    def init_window_size_font_size(self):
        # reset the window size, line thickness, font scale if the width of the first image
        # in the image path list is greater than 1000.
        # Here we assume all the images in the directory has the same size, if this is not the
        # case, we may need to design more advanced setting (load all images, or sampling 5 images from
        # the image path list, and then use the maximum size of the image)
        if len(self.IMAGE_PATH_LIST) > 0:
            image_path = self.IMAGE_PATH_LIST[0]
            tmp_image = cv2.imread(image_path)
            assert tmp_image is not None

            height, width, _ = tmp_image.shape
            # print(f'image size: height={height}, width={width} for {os.path.dirname(image_path)}')  # (400, 640, 3)

            # change the setting for the window and line thickness if the image width > 1000
            if width > 1000:
                # self.window_width = width + 50  # 1000
                # self.window_height = height + 50  # 700
                self.LINE_THICKNESS = 2
                self.window_width = 1300  # 1000
                self.window_height = 800  # 700
                self.class_name_font_scale = 0.8
            else:
                self.LINE_THICKNESS = 1
                self.window_width = 1000
                self.window_height = 700
                self.class_name_font_scale = 0.6

    # initialize the class list, last class index, and class index
    def init_class_list_and_class_color(self):
        if not os.path.isfile(self.class_list_file):
            print('File {} not exist, please check.'.format(self.class_list_file))
            sys.exit(0)

        # load class list
        with open(self.class_list_file) as f:
            self.CLASS_LIST = list(smrc.utils.non_blank_lines(f))
        # print(self.CLASS_LIST)

        self.LAST_CLASS_INDEX = len(self.CLASS_LIST) - 1

        # the final definitions of the colors of the classes for bbox plotting
        # make sure we have defined enough class colors for the loaded class names.
        self.CLASS_BGR_COLORS = self.init_class_color(self.CLASS_LIST)

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

    def set_class_index(self, ind):
        self.active_class_index = ind
        text = 'Selected class {}/{} -> {}'.format(self.active_class_index, self.LAST_CLASS_INDEX,
                                                   self.CLASS_LIST[self.active_class_index])
        self.display_text(text, 3000)

    def set_active_bbox_idx_based_on_mouse_position(self, allow_none=True):
        # print('self.mouse_x = {}, self.mouse_y = {}'.format(self.mouse_x, self.mouse_y))
        selected_bbox_inx = None  # local variable, not global variable

        smallest_area = -1
        # if clicked inside multiple bboxes selects the smallest one
        for idx, obj in enumerate(self.active_image_annotated_bboxes):
            _, x1, y1, x2, y2 = obj
            xmin, ymin, xmax, ymax = self.get_bbox_active_region_rectangle(x1, y1, x2, y2)

            if smrc.utils.point_in_rectangle(self.mouse_x, self.mouse_y, xmin, ymin, xmax, ymax):
                tmp_area = smrc.utils.get_bbox_area(xmin, ymin, xmax, ymax)
                if tmp_area < smallest_area or smallest_area == -1:
                    smallest_area = tmp_area
                    selected_bbox_inx = idx
        #
        if allow_none:
            self.active_bbox_idx = selected_bbox_inx
        elif not allow_none and selected_bbox_inx is not None:
            self.active_bbox_idx = selected_bbox_inx

    def set_active_anchor_position(self):
        self.active_anchor_position = None

        # print('self.active_bbox_idx =', self.active_bbox_idx)
        if self.is_valid_active_bbox():
            eX, eY = self.mouse_x, self.mouse_y
            _, x_left, y_top, x_right, y_bottom = self.active_image_annotated_bboxes[self.active_bbox_idx]

            # if mouse cursor is noe inside the inner boundaries of region of n active bbox
            if smrc.utils.point_in_rectangle(eX, eY,
                           x_left - self.BBOX_ANCHOR_THICKNESS,
                           y_top - self.BBOX_ANCHOR_THICKNESS,
                           x_right + self.BBOX_ANCHOR_THICKNESS,
                           y_bottom + self.BBOX_ANCHOR_THICKNESS) \
                    and (not smrc.utils.point_in_rectangle(eX, eY,
                           x_left + self.BBOX_ANCHOR_THICKNESS,
                           y_top + self.BBOX_ANCHOR_THICKNESS,
                           x_right - self.BBOX_ANCHOR_THICKNESS,
                           y_bottom - self.BBOX_ANCHOR_THICKNESS)):
                # first row: horizontal, second row: vertical, of the center of the 8 anchor rectangles
                end_points = np.array(
                    [[x_left, x_left, (x_left + x_right) / 2, (x_left + x_right) / 2, x_right],  # horizontal
                     [y_top, y_top, (y_top + y_bottom) / 2, (y_top + y_bottom) / 2, y_bottom]])

                # print('end_points =', end_points)
                # left shift, right shift of the center of the anchor rectangle
                left_right_shift = np.array([-1, 1, -1, 1, -1]) * self.BBOX_ANCHOR_THICKNESS
                # print('left_right_shift =', left_right_shift)
                # np.array([[eX], [eY]]) is the mouse_cursor_position

                end_points = end_points + left_right_shift
                # print('end_points =', end_points)

                # print('mouse_curse =', np.array([[eX], [eY]]))
                position_indicators = np.array([[eX], [eY]]) - end_points

                # print('position_indicators =', position_indicators)
                indices = np.sum(position_indicators >= 0, axis=1) - 1  # axis=1, sum over the horizontal direction
                # print('indices =', indices)

                self.active_anchor_position = self.ANCHOR_POSITION_ENCODING[0][indices[0]] + \
                                              self.ANCHOR_POSITION_ENCODING[1][indices[1]]

    def set_active_bbox_idx_if_NONE(self):
        if self.active_bbox_to_set is not None:
            if self.active_bbox_to_set in self.active_image_annotated_bboxes:
                self.active_bbox_idx = self.active_image_annotated_bboxes.index(
                    self.active_bbox_to_set)
            self.active_bbox_to_set = None

        if self.active_bbox_idx is None:  # we do nothing if self.active_bbox_idx is already set
            if len(self.active_image_annotated_bboxes) > 0:  # if we have any annotated bbox
                self.active_bbox_idx = 0  # initialize the self.active_bbox_idx

                # there is more than one annotated box and active_bbox_previous_image was set
                # set the active bbox based on the active_bbox in previous image
                if len(self.active_image_annotated_bboxes) > 1 and \
                        self.active_bbox_previous_image is not None:
                    # find the bbox in the current image that is closest to active_bbox_previous_image
                    smallest_distance = np.linalg.norm(
                            np.asarray([self.active_image_width, self.active_image_height,
                                        self.active_image_width, self.active_image_height]) - np.asarray([0, 0, 0, 0])
                        )
                    #print('self.active_image_annotated_bboxes', self.active_image_annotated_bboxes)
                    #print('self.active_bbox_previous_image', self.active_bbox_previous_image)
                    for idx, bbox in enumerate(self.active_image_annotated_bboxes):
                        #print('idx =', idx, 'bbox =', bbox)
                        dist = np.linalg.norm(
                            np.asarray(bbox[1:]) - np.asarray(self.active_bbox_previous_image[1:])
                        )
                        #print('dist =', dist)
                        if dist < smallest_distance:
                            self.active_bbox_idx = idx
                            smallest_distance = dist
                            #print('self.active_bbox_idx =',self.active_bbox_idx,
                            #      'smallest_distance =', smallest_distance)

    # draw horizontal and vertical lines that across the cursor point
    def draw_line(self, tmp_img, x, y, height, width, color, line_thickness):
        # cv2.line(tmp_img, (x, 0), (x, height), color, line_thickness)
        # cv2.line(tmp_img, (0, y), (width, y), color, line_thickness)
        length = 20
        cv2.line(tmp_img, (x, y - length), (x, y + length), color, line_thickness)
        cv2.line(tmp_img, (x - length, y), (x + length, y), color, line_thickness)
        # cv2.drawMarker(img, (x, y), color, cv2.MARKER_CROSS, 15, 2) #

        # print(f'self.CLASS_LIST = {self.CLASS_LIST}, self.active_class_index = {self.active_class_index}')
        name_of_class = self.CLASS_LIST[self.active_class_index]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        margin = 3
        text_width, text_height = cv2.getTextSize(name_of_class, font, font_scale, line_thickness)[0]
        cv2.rectangle(tmp_img,
                      (self.mouse_x + line_thickness, self.mouse_y - line_thickness),
                      (self.mouse_x + text_width + margin, self.mouse_y - text_height - margin),
                      color, -1)  # complement_bgr(color)
        cv2.putText(tmp_img, name_of_class, (self.mouse_x + margin, self.mouse_y - margin),
                    font, font_scale, self.ACTIVE_BBOX_COLOR, line_thickness,
                    cv2.LINE_AA)  #

    # draw the active anchor when mouse is moving around the dragging regions of the active bbox
    # of the current operating image
    def draw_active_anchor(self, tmp_img):
        if self.active_anchor_position is not None:
            if self.is_valid_active_bbox():
                active_bbox = self.active_image_annotated_bboxes[self.active_bbox_idx]
                self.draw_active_bbox_active_anchor(
                    tmp_img, active_bbox, active_anchor_position=self.active_anchor_position,
                    pt=(self.mouse_x, self.mouse_y))

    # draw the active anchor when mouse is moving around the dragging regions of the active bbox
    # of the current operating image
    def draw_active_bbox_active_anchor(self, tmp_img, active_bbox, active_anchor_position, pt):
        line_thickness = self.LINE_THICKNESS
        arrow_magnitude = self.active_anchor_arrow_magnitude
        multiplier = 1

        _, xmin, ymin, xmax, ymax = active_bbox
        if smrc.utils.get_bbox_area(xmin, ymin, xmax, ymax) > self.bbox_area_threshold_for_thick_line:
            line_thickness = line_thickness * 2
            multiplier = 1.2  # modify the multiplier so that large bbox get large arrow

        arrow_shifts = self.ACTIVE_ANCHOR_DICTIONARY[active_anchor_position]  # return a list of points (x,y)

        if arrow_shifts is not None:
            for point_shift in arrow_shifts:
                pt1 = (
                    int(pt[0] + point_shift[0] * multiplier),
                    int(pt[1] + point_shift[1] * multiplier)
                )

                smrc.utils.draw_arrow(tmp_img, pt, pt1, (255, 255, 255),
                                      int(arrow_magnitude * multiplier), line_thickness)

    def draw_bboxes_from_file(self, tmp_img, ann_path):
        '''
        # load the draw bbox from file, and initialize the annotated bbox in this image
        for fast accessing (annotation -> self.active_image_annotated_bboxes)
            ann_path = labels/489402/0000.txt
            print('this ann_path =', ann_path)
        '''
        self.active_image_annotated_bboxes = []  # initialize the image_annotated_bboxes
        if os.path.isfile(ann_path):
            with open(ann_path, 'r') as old_file:
                lines = old_file.readlines()
            old_file.close()

            for line in lines:
                result = line.split(' ')

                # the data format in line (or txt file) should be int type, 0-index.
                # we transfer them to int again even they are already in int format (just in case they are not)
                bbox = [int(result[0]), int(result[1]), int(result[2]), int(
                    result[3]), int(result[4])]

                self.active_image_annotated_bboxes.append(bbox)
                self.draw_annotated_bbox(tmp_img, bbox)

    def draw_active_bbox(self, tmp_img):
        # self.active_bbox_idx < len(self.active_image_annotated_bboxes) is neccessary, otherwise, it cuases error
        # when we changing the image frame in a very fast speed (dragging trackbar) so that setting acitve bbox
        # is not finished
        if self.is_valid_active_bbox():
            # do not change the class_index here, otherwise every time the active_bbox_idx changed (mouse is wandering),
            # the class_index will change (this is not what we want).
            # the data format should be int type, class_idx is 0-index.
            bbox = self.active_image_annotated_bboxes[self.active_bbox_idx]

            self.draw_single_bbox(tmp_img, bbox, self.ACTIVE_BBOX_COLOR,
                                  self.ACTIVE_BBOX_COLOR,
                                  self.ACTIVE_BBOX_COLOR)
            # draw_single_bbox(self, tmp_img, bbox, rectangle_color, anchor_rect_color, text_shadow_color)

    def draw_last_added_bbox(self, tmp_img):
        if self.last_added_bbox is not None:
            # Do not change the class_index here, otherwise every time the active_bbox_idx changed (mouse is wandering),
            # the class_index will change (this is not what we want).
            bbox = self.last_added_bbox

            self.draw_single_bbox(tmp_img, bbox, smrc.utils.complement_bgr(self.ACTIVE_BBOX_COLOR),
                                  smrc.utils.complement_bgr(self.ACTIVE_BBOX_COLOR),
                                  smrc.utils.complement_bgr(self.ACTIVE_BBOX_COLOR))
            # draw_single_bbox(self, tmp_img, bbox, rectangle_color, anchor_rect_color, text_shadow_color)

    def draw_changed_label_bbox(self, tmp_img):
        bbox = self.active_image_annotated_bboxes[self.active_bbox_idx]

        self.draw_single_bbox(tmp_img, bbox, smrc.utils.complement_bgr(self.ACTIVE_BBOX_COLOR),
                              smrc.utils.complement_bgr(self.ACTIVE_BBOX_COLOR),
                              self.ACTIVE_BBOX_COLOR)


# class ImageSequenceViewer:


class AnnotationTool(MainWindow):
    def __init__(self, image_dir=None, label_dir=None, class_list_file=None):
        # super(MainWindow).__init__()
        MainWindow.__init__(self)
        self.keyboard_encoder = KeyboardCoding()
        self.keyboard = self.keyboard_encoder.keyboard

        self.IMAGE_DIR = image_dir
        self.LABEL_DIR = label_dir
        self.class_list_file = class_list_file

        self.ANNOTATION_FORMAT = '.txt'  # 'YOLO_darknet', 'PASCAL_VOC' : '.xml',
        self.active_directory = None  # the name of the directory we are annotating
        # the directory where the final annotation results are moved to
        self.ANNOTATION_FINAL_DIR = None  # 'annotation_finished' 

        # for operations
        self.edges_on = False  # indicate if we show the edges on the image
        self.dragging_on = False  # indicate if the dragging operation is enabled
        self.label_changed_flag = False  # indicate a label is just been changed.

        self.moving_on = False
        # the minimum pixel unit each time we pressed -->, <--, upward, downward to move the active bbox
        self.move_unit = 2

        self.anchor_being_dragged = None  # indicate if an anchor is being dragged
        # record the position of the mouse cursor pointer if EVENT_LBUTTONDOWN is Triggered
        # self.previously_pressed_key = None
        self.initial_dragging_position = None

        # record the last added bbox automatically for fast pasting
        self.display_last_added_bbox_on = False
        # a list of dict, each dict record the key (ann_path), and value (bbox_list)
        # of the deleted bbox for undo delete
        self.deleted_bbox_history = []

        # key: image_idx, value: bbox_idx (we do not save the bbox because
        # the user may modify the bbox (e.g., moving, dragging) during the fitting process
        self.curve_fitting_dict = {}
        # delay the curve fitting operation when adjusting a bbox
        # self.curve_fitting_dict_updated = False
        # record the last curve-fitted bboxes for undo curve fitting
        self.last_fitted_bbox = []
        self.fitted_bbox = []  # format, a list of [image_id, bbox]
        self.fitting_mode_on = False
        self.curve_fitting_overlap_suppression_thd = 0.50

        self.overlap_suppression_thd = 0.65  # suppressing the highly overlapped bbox
        # indicate if we quit the annotation tool
        self.quit_annotation_tool = False

        if self.class_list_file is not None and os.path.isfile(self.class_list_file):
            self.init_class_list_and_class_color()

        self.active_bbox_to_set = None
        # print(f'self.class_list_file = {self.class_list_file}')
        # # print(os.path.isfile(self.class_list_file))

    def get_annotation_path(self, img_path):
        new_path = img_path.replace(self.IMAGE_DIR, self.LABEL_DIR, 1)
        _, img_ext = os.path.splitext(new_path)
        annotation_path = new_path.replace(img_ext, self.ANNOTATION_FORMAT, 1)
        # print(annotation_path) #output/0000.txt
        return annotation_path

    def get_image_path_from_annotation_path(self, txt_file_name):
        return smrc.utils.get_image_or_annotation_path(
            txt_file_name, self.LABEL_DIR,
            self.IMAGE_DIR, '.jpg'
        )

    def reset_after_image_change(self):
        # avoid mis-operation if the user move to previous or next image when dragging a bbox
        self.dragging_on = False
        self.moving_on = False

        self.point_1, self.point_2 = None, None
        self.active_anchor_position = None

        # every time we change the image frame, we automatically record the active bbox if the previously operating
        # image frame
        if self.is_valid_active_bbox():
            # print('self.active_bbox_idx  =', self.active_bbox_idx)
            self.active_bbox_previous_image = self.active_image_annotated_bboxes[self.active_bbox_idx]

        self.active_bbox_idx = None
        # if self.game_controller_available and self.game_controller_on:
        # print('self.active_bbox_previous_image = ', self.active_bbox_previous_image)

    ########################################################
    # for annotating the selected directory
    ########################################################

    def load_active_directory(self):
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
            test_img = cv2.imread(f_path)
            if test_img is not None:
                self.IMAGE_PATH_LIST.append(f_path)
        self.LAST_IMAGE_INDEX = len(self.IMAGE_PATH_LIST) - 1
        if self.LABEL_DIR is not None:
            labels_dir = os.path.join(self.LABEL_DIR, self.active_directory)
            # make the corresponding label dir if it does not exist
            if not os.path.exists(labels_dir):
                os.makedirs(labels_dir)

        # # create empty annotation files for each image, if it doesn't exist already
        # for img_path in self.IMAGE_PATH_LIST:
        #     ann_path = self.get_annotation_path(img_path)
        #     # print('ann_path = ', ann_path)
        #     if not os.path.isfile(ann_path):
        #         open(ann_path, 'a').close()

        self.load_active_directory_additional_operation()

    def load_active_directory_additional_operation(self):
        pass

    def delete_active_bbox(self):
        if self.is_valid_active_bbox():
            active_bbox = self.get_active_bbox()
            if active_bbox is not None and self.fitting_mode_on and \
                self.active_image_index in self.curve_fitting_dict and \
                    self.curve_fitting_dict[self.active_image_index] == active_bbox:

                del self.curve_fitting_dict[self.active_image_index]
                    # self.curve_fit_manually()

            self.delete_any_bbox_by_idx(self.active_bbox_idx)
            # print('active_bbox_idx=', self. active_bbox_idx)

            self.delete_active_bbox_additional()
        # record this delete into history for later recovery use
        # ann_path = self.active_image_annotation_path
        # self.add_single_bbox_delete_history(
        #     ann_path, self.active_image_annotated_bboxes[self.active_bbox_idx]
        # )
        # del self.active_image_annotated_bboxes[self.active_bbox_idx]
        # smrc.line.save_bbox_to_file(ann_path, self.active_image_annotated_bboxes)

    def delete_active_bbox_additional(self):
        """
        additional operation after delete active bbox
        :return:
        """
        pass

    def delete_any_bbox_by_idx(self, bbox_idx):
        """
        all delete operation for active image by bbox idx should be handled by
        this function
        :param bbox_idx: the idx of the bbox to delete in the bbox_list of the
        active image (self.active_image_annotated_bboxes)
        :return:
        """
        # record this delete into history for later recovery use
        ann_path = self.active_image_annotation_path
        self.add_delete_history_for_single_bbox(
            ann_path, [self.active_image_annotated_bboxes[bbox_idx]]
        )
        smrc.utils.delete_any_specified_bbox_idx(
            ann_path, self.active_image_annotated_bboxes, bbox_idx
        )
        # del self.active_image_annotated_bboxes[bbox_idx]
        # smrc.line.save_bbox_to_file(ann_path, self.active_image_annotated_bboxes)

    def add_single_bbox(self, ann_path, bbox):
        """
        all bbox adding operation for any annotation path should be handled by
        this function
        :param ann_path:
        :param bbox:
        :return:
        """
        adding_result = 0
        if bbox in self.active_image_annotated_bboxes:
            text_content = f'This bounding box {bbox} has been added already.'
            self.display_text(text_content, 1000)
            # self.display_text_on_image(tmp_img,text_content)
        else:
            smrc.utils.save_bbox_to_file_incrementally(ann_path, [bbox])
            # txt_line = smrc.line.bbox_transfer_to_txt_line_format(class_idx, xmin, ymin, xmax, ymax)
            # with open(ann_path, 'a') as myfile:
            #     myfile.write(txt_line + '\n')  # append line
            # myfile.close()

            adding_result = 1
            self.active_image_annotated_bboxes.append(bbox)
            self.last_added_bbox = bbox
        # ===================================================check in future
        # # whatever adding this bbox succeed or failed, we return it back
        # self.last_added_bbox = bbox
        # ===================================================
        return adding_result

    def add_bbox(self, ann_path, class_idx, p1, p2):
        """
        adding bbox after a bbox is drew (p1, p2 given)
        :param ann_path: annotation path
        :param class_idx: class index
        :param p1: first vertex of a rectangle
        :param p2: second vertex
        :return:
        """
        xmin, ymin = min(p1[0], p2[0]), min(p1[1], p2[1])
        xmax, ymax = max(p1[0], p2[0]), max(p1[1], p2[1])
        bbox = [class_idx, xmin, ymin, xmax, ymax]
        adding_result = self.add_single_bbox(ann_path, bbox)

        if self.fitting_mode_on and adding_result:
            self.curve_fitting_dict[self.active_image_index] = bbox
            print(f'{bbox} added into {self.active_image_index} ')

        self.add_bbox_additional()
        return adding_result, bbox

    def add_bbox_additional(self):
        """
        # addtional operation after adding a bbox
        :return:
        """
        pass

    def update_active_bbox(self, new_bbox):
        adding_result = 0
        if self.is_valid_active_bbox():
            self.active_image_annotated_bboxes[self.active_bbox_idx] = new_bbox
            smrc.utils.save_bbox_to_file(self.active_image_annotation_path, self.active_image_annotated_bboxes)
            adding_result = 1
        self.last_added_bbox = new_bbox

        # for curve_fitting
        if self.fitting_mode_on:
            # directly fitting after update is very slow when dragging a bbox
            self.curve_fitting_dict[self.active_image_index] = new_bbox
            # self.curve_fit_manually()
            # self.curve_fitting_dict_updated = True
        return adding_result

    def add_delete_history_for_single_bbox(self, ann_path, bbox):
        deleted_bbox = {
            ann_path: bbox
        }
        self.deleted_bbox_history.append(deleted_bbox)

    def undo_last_delete(self):
        """
        unconditional recover the latest delete
        :return:
        """
        undo_succeed = False
        last_deleted_bbox_dict = self.deleted_bbox_history[-1]
        for ann_path in last_deleted_bbox_dict:
            bbox_list = last_deleted_bbox_dict[ann_path]
            print(f'undo delete {ann_path} bbox {bbox_list}')
            smrc.utils.save_bbox_to_file_incrementally(ann_path=ann_path, bbox_list=bbox_list)

        # update the delete history
        del self.deleted_bbox_history[-1]
        return undo_succeed

    def undo_delete_single_bbox(self):
        undo_succeed = False
        if len(self.deleted_bbox_history) == 0:
            print('No bounding bbox to recover.')
            return undo_succeed
            # self.display_temp_text_on_image(No)

        # image_path = self.IMAGE_PATH_LIST[self.active_image_index]
        ann_path = self.active_image_annotation_path

        last_deleted_bbox_dict = self.deleted_bbox_history[-1]
        if len(last_deleted_bbox_dict) == 1 and \
                ann_path in last_deleted_bbox_dict:
            undo_succeed = self.undo_last_delete()

            self.undo_delete_single_bbox_additional_operation()
            return undo_succeed

    def undo_delete_single_bbox_additional_operation(self):
        pass

    def undo_delete_single_tracklet(self):
        undo_succeed = False
        if len(self.deleted_bbox_history) == 0:
            print('No bounding bbox to recover.')
            return undo_succeed
            # self.display_temp_text_on_image(No)

        # image_path = self.IMAGE_PATH_LIST[self.active_image_index]
        ann_path = self.active_image_annotation_path
        last_deleted_bbox_dict = self.deleted_bbox_history[-1]
        if len(last_deleted_bbox_dict) >= 1 and \
                ann_path in last_deleted_bbox_dict:
            undo_succeed = self.undo_last_delete()
            return undo_succeed

    def update_active_bbox_label(self, target_class_idx):
        if self.is_valid_active_bbox():
            self.active_image_annotated_bboxes[self.active_bbox_idx][0] = target_class_idx
            smrc.utils.save_bbox_to_file(self.active_image_annotation_path, self.active_image_annotated_bboxes)

    def update_active_bbox_boundaries(self, xmin_new, ymin_new, xmax_new, ymax_new):
        if self.is_valid_active_bbox():
            class_idx = self.active_image_annotated_bboxes[self.active_bbox_idx][0]
            new_bbox = [class_idx, xmin_new, ymin_new, xmax_new, ymax_new]
            self.update_active_bbox(new_bbox)

    # conduct bbox translation for the active bbox (move the bbox using the mouse)
    # T = (move_in_x_direction, move_in_y_direction)
    def translate_active_bbox(self, T):
        """
        translate the center of a bbox, x0, y0
        :param T:
        :return:
        """
        if self.is_valid_active_bbox():
            class_idx, xmin, ymin, xmax, ymax = self.active_image_annotated_bboxes[self.active_bbox_idx]

            # do nothing if the translation is illegal (i.e., the bbox is out of the image region
            # [image_width * image_height])
            if xmin + T[0] < 0 or xmax + T[0] >= self.active_image_width or \
                    ymin + T[1] < 0 or ymax + T[1] >= self.active_image_height:
                print('We do nothing because the operation causes the bbox moving out of the image region.')
            else:
                xmin, xmax = xmin + T[0], xmax + T[0]
                ymin, ymax = ymin + T[1], ymax + T[1]
                new_bbox = [class_idx, xmin, ymin, xmax, ymax]
                #print(new_bbox)
                self.update_active_bbox(new_bbox)

    def enlarge_active_bbox(self, wt, ht):
        """
        if wt is negative, then the operation is to shrink the active bbox.
        The some comment applies to ht.
        :param wt:
        :param ht:
        :return:
        """
        if self.is_valid_active_bbox():
            class_idx, xmin, ymin, xmax, ymax = self.active_image_annotated_bboxes[self.active_bbox_idx]

            # Here we do not need to warry about the bbox is moving out of the image region
            # as only of the four boundaries is operating.

            if 0 <= xmin - wt < self.active_image_width:  # left
                xmin = xmin - wt

            if 0 <= xmax + wt < self.active_image_width:  # right
                xmax = xmax + wt

            if 0 <= ymin - ht < self.active_image_height:  # top
                ymin = ymin - ht

            if 0 <= ymax + ht < self.active_image_height:  # bottom
                ymax = ymax + ht
            x1, x2 = min(xmin, xmax), max(xmin, xmax)
            y1, y2 = min(ymin, ymax), max(ymin, ymax)
            new_bbox = [class_idx, x1, y1, x2, y2]

            self.update_active_bbox(new_bbox)

    def translate_active_bbox_boundary(self, T):
        """
        translate one of the 4 boundaries of a bbox, x1 or y1 or x2 or y2s
        format of T, [move_direction, move_value], move_direction = 'top', 'bottom', 'left', 'right'
        # 4 degrees of freedom
        :param T:
        :return:
        """
        if self.is_valid_active_bbox():
            class_idx, xmin, ymin, xmax, ymax = self.active_image_annotated_bboxes[self.active_bbox_idx]

            # Here we do not need to warry about the bbox is moving out of the image region
            # as only of the four boundaries is operating.
            if T[0] == 'left' and 0 <= xmin + T[1] < self.active_image_width:
                xmin = xmin + T[1]

            elif T[0] == 'top' and 0 <= ymin + T[1] < self.active_image_height:
                ymin = ymin + T[1]

            elif T[0] == 'right' and 0 <= xmax + T[1] < self.active_image_width:
                xmax = xmax + T[1]

            elif T[0] == 'bottom' and 0 <= ymax + T[1] < self.active_image_height:
                ymax = ymax + T[1]

            new_bbox = [class_idx, xmin, ymin, xmax, ymax]

            self.update_active_bbox(new_bbox)

    # // need to improve this function
    def post_process_bbox_coordinate(self, xmin, ymin, xmax, ymax):
        if xmin > xmax or ymin > ymax:
            xmin, ymin = min(xmin, xmax), min(ymin, ymax)
            xmax, ymax = max(xmin, xmax), max(ymin, ymax)
            print('????????????????????????????????????????? What? xmin >= xmax, ymin >= ymax')
        elif xmin == xmax:
            xmax = xmin + 1
        elif ymin == ymax:
            ymax = ymin + 1

        if 0 <= min(xmin, xmax) < self.active_image_width and \
                0 <= max(xmin, xmax) < self.active_image_width and \
                0 <= min(ymin, ymax) < self.active_image_height and \
                0 <= max(ymin, ymax) < self.active_image_height:
            return [xmin, ymin, xmax, ymax]
        else:
            if xmin < 0 or xmax < 0 or ymin < 0 or ymax < 0:
                if xmin < 0:
                    xmin = 0
                if xmax < 0:
                    xmax = 0
                if ymin < 0:
                    ymin = 0
                if ymax < 0:
                    ymax = 0

            if xmin >= self.active_image_width or xmax >= self.active_image_width or \
                ymin >= self.active_image_height or ymax >= self.active_image_height:

                if xmin >= self.active_image_width:
                    xmin = self.active_image_width - 1

                if xmax >= self.active_image_width:
                    xmax = self.active_image_width - 1

                if ymin >= self.active_image_height:
                    ymin = self.active_image_height - 1

                if ymax >= self.active_image_height:
                    ymax = self.active_image_height - 1
            return [xmin, ymin, xmax, ymax]

    # update the active bbox based on the dragging operation of the user
    def update_dragging_active_bbox(self):
        if self.is_valid_active_bbox() and self.active_anchor_position is not None:

            # the position of the mouse cursor
            eX, eY = self.mouse_x, self.mouse_y

            # moving the center of the active bbox
            if self.active_anchor_position[0] == "O" or self.active_anchor_position[1] == "O":
                if self.initial_dragging_position is not None:
                    #print('left click, EVENT_LBUTTONDOWN, initial_dragging_position = ', self.initial_dragging_position)
                    T = (eX - self.initial_dragging_position[0], eY - self.initial_dragging_position[1])
                    #print('T = ', T )
                    self.translate_active_bbox(T)

                    #we need to update the initial_dragging_position when the mouse keeps moving.
                    #otherwise, the T is actually acculating all the translations
                    self.initial_dragging_position = (self.mouse_x, self.mouse_y)
            # moving one of the four boundaries of the active bbox
            else:
                _class_idx, x_left, y_top, x_right, y_bottom = self.active_image_annotated_bboxes[self.active_bbox_idx]
                # Do not allow the bbox to flip upside down (given a margin)
                margin = 3 * self.BBOX_ANCHOR_THICKNESS
                change_was_made = False
                if self.active_anchor_position[0] == "L":
                    # left anchors (LT, LM, LB)
                    if eX < x_right - margin:
                        x_left = eX
                        change_was_made = True
                elif self.active_anchor_position[0] == "R":
                    # right anchors (RT, RM, RB)
                    if eX > x_left + margin:
                        x_right = eX
                        change_was_made = True
                if self.active_anchor_position[1] == "T":
                    # top anchors (LT, RT, MT)
                    if eY < y_bottom - margin:
                        y_top = eY
                        change_was_made = True
                elif self.active_anchor_position[1] == "B":
                    # bottom anchors (LB, RB, MB)
                    if eY > y_top + margin:
                        y_bottom = eY
                        change_was_made = True

                x_left, y_top, x_right, y_bottom = self.post_process_bbox_coordinate(x_left, y_top, x_right, y_bottom)

                if change_was_made:
                    self.update_active_bbox_boundaries(x_left, y_top, x_right, y_bottom)

    def copy_active_bbox(self):
        active_bbox = self.get_active_bbox()
        if active_bbox is not None:
            self.last_added_bbox = active_bbox
            self.display_last_added_bbox_on = True

    def paste_last_added_bbox(self):
        if self.last_added_bbox is not None:
            class_idx, xmin, ymin, xmax, ymax = self.last_added_bbox

            # add self.last_added_bbox and update it to the latest added bbox
            self.add_bbox(self.active_image_annotation_path, class_idx, (xmin, ymin),
                          (xmax, ymax))

            # get the newly pasted bbox ready for being moved
            if self.last_added_bbox in self.active_image_annotated_bboxes:
                self.active_bbox_idx = self.active_image_annotated_bboxes.index(self.last_added_bbox)
                self.moving_on = True

    def read_pressed_key(self):
        return self.keyboard_encoder.read_pressed_key()


class AnnotationToolDeprecated(AnnotationTool):
    def __init__(self):
        super().__init__()

    @staticmethod
    def delete_active_bbox_non_max_suppression(compress_thd):
        """
        this function should be rewrite,
        :param compress_thd:
        :return:
        """
        print('delete_active_bbox_non_max_suppression should be rewrite')
        pass
        #
        # candidate_bbox_list = []
        # candidate_class_id_list = []
        # if self.is_valid_active_bbox():
        #     class_idx_active, xmin, ymin, xmax, ymax = self.active_image_annotated_bboxes[self.active_bbox_idx]
        #     active_bbox_rect = [xmin, ymin, xmax, ymax]
        #     candidate_class_id_list.append(class_idx_active) #add the class of active bbox to the list
        #     candidate_bbox_list.append([class_idx_active, xmin, ymin, xmax, ymax])  #add the active bbox to the list
        #
        #     print('candidate_bbox_list = ', candidate_bbox_list)
        #     for idx, bbox in enumerate(self.active_image_annotated_bboxes):
        #         if idx != self.active_bbox_idx:
        #             class_idx, x1, y1, x2, y2 = bbox
        #             print([x1, y1, x2, y2])
        #             print('iou = ', smrc.line.compute_iou(active_bbox_rect, [x1, y1, x2, y2]))
        #             if smrc.line.compute_iou(active_bbox_rect, [x1, y1, x2, y2]) > compress_thd:
        #                 candidate_bbox_list.append([class_idx, x1, y1, x2, y2])
        #                 candidate_class_id_list.append(class_idx) # add the class of non active bbox to the list
        #
        #     print('candidate_bbox_list = ', candidate_bbox_list)
        #     if len(candidate_bbox_list) > 1:  # more than one non_active_bbox
        #         # if self.active_class_index in candidate_class_id_list:
        #         #     # A.index(b), get the first element in the list A of which value is b
        #         #     keep_bbox_idx = candidate_class_id_list.index(self.active_class_index)
        #         # else:
        #         #     # keep the active bbox id, if you want to delete the active bbox, you can use right mouse button
        #         #     keep_bbox_idx = 0
        #
        #         keep_bbox_idx = 0
        #         print('keep_bbox_idx = ', keep_bbox_idx)
        #         # delete the bbox of which class is not the self.active_class_index
        #         for idx, bbox in enumerate(candidate_bbox_list):
        #             if idx != keep_bbox_idx:
        #                 box_id_to_delete = self.active_image_annotated_bboxes.index(bbox)
        #                 self.delete_any_bbox_by_idx(box_id_to_delete)


