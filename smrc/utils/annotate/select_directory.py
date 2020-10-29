import cv2
import sys
import pygame

from .annotation_tool import AnnotationTool
from .game_controller import GameController
from .keyboard import KeyboardCoding
import smrc.utils


class SelectDirectory(AnnotationTool,  GameController, KeyboardCoding):
    def __init__(self, directory_list=None, directory_list_file=None):
        AnnotationTool.__init__(self)
        GameController.__init__(self)
        KeyboardCoding.__init__(self)

        # modify the window name for the application
        self.IMAGE_WINDOW_NAME = 'SelectingDirectory'

        # used by selecting directory tool
        self.TRACKBAR_DIRECTORY = 'DirectoryPage'

        self.DIRECTORY_LIST = []
        
        # We will edit this file automatically every time one annotation is done.
        self.directory_list_file = directory_list_file

        # Make sure we have at least one directory to select
        if directory_list_file is None:
            if directory_list is None or len(directory_list) == 0:
                print('You have to specify the directory_list.')
                sys.exit(0)
            else:
                self.DIRECTORY_LIST = directory_list

        # the name of the directory we choose to annotate
        self.active_directory = None  

        # font size for the directory name to display
        self.directory_name_font_scale = 1
        self.line_thickness = 1
        
        # Record the (active) directory that the current user is operating, as defined in AnnotationTool()
        # If user_name not specified, then it will be 'active_directory.txt'
        # Currently not used.
        self.active_directory_file = None 

        # the directories in the current windows if there are multiple pages
        self.directory_list_current_page = []
        self.LAST_DIRECTORY_PAGE_INDEX = None
    
        # design the display format of the directory list in the image window
        self.page_max_num_cols = 5
        self.directory_number_per_page = 20
        self.directory_page_index = 0  # the index of the directory page
        # self.margin = 3  # margin from the bbox boundary to the font 

        # if self.user_name is None or len(self.user_name) == 0:
        #     self.directory_list_file = 'directory_list.txt'
        #     self.active_directory_file = 'active_directory.txt'
            
        #     # generate directory_list.txt automatically
        #     if self.auto_load_directory in ['image', 'label']:
        #         dir_list = []
        #         if self.auto_load_directory == 'image':
        #             dir_list = smrc.line.get_dir_list_in_directory(
        #                 self.IMAGE_DIR
        #                 ) 
        #         elif self.auto_load_directory == 'label':
        #             dir_list = smrc.line.get_dir_list_in_directory(
        #                 self.IMAGE_DIR
        #                 ) 
        #         smrc.line.save_1d_list_to_file(self.directory_list_file, dir_list)
        # else:
        #     self.user_name = self.user_name.lower()
        #     self.directory_list_file = 'directory_list_' + self.user_name + '.txt'
        #     self.active_directory_file = 'active_directory_' + self.user_name + '.txt'

        # # create the directory or file if they do not exist
        # if not os.path.isfile(self.directory_list_file):
        #     open(self.directory_list_file, 'a').close()
        # if not os.path.isfile(self.active_directory_file):
        #     open(self.active_directory_file, 'a').close()

    def init_image_window_and_mouse_listener(self):
        # create window
        cv2.namedWindow(self.IMAGE_WINDOW_NAME, cv2.WINDOW_KEEPRATIO)  # cv2.WINDOW_KEEPRATIO cv2.WINDOW_AUTOSIZE
        cv2.resizeWindow(self.IMAGE_WINDOW_NAME, self.window_width, self.window_height)
        cv2.setMouseCallback(self.IMAGE_WINDOW_NAME, self.mouse_listener_for_image_window)
    
    def mouse_listener_for_image_window(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_x = x
            self.mouse_y = y
            # print('mouse move,  EVENT_MOUSEMOVE')

        elif event == cv2.EVENT_LBUTTONDOWN:
            # print('left click, EVENT_LBUTTONDOWN')
            if self.active_bbox_idx is not None and \
                self.directory_list_current_page is not None and \
                    self.active_bbox_idx < len(self.directory_list_current_page):
                self.active_directory = self.directory_list_current_page[self.active_bbox_idx]
                # print('self.active_directory =', self.active_directory )

    # which directories to load are based on the value of self.directory_page_index, self.DIRECTORY_LIST
    # self.active_image_annotated_bboxes, self.directory_exist_list and self.directory_list_current_page are loaded
    def load_directory_name_for_current_page(self, directory_img):
        margin = 3  # margin from the bbox boundary to the font
        if len(self.DIRECTORY_LIST) == 0:
            if self.directory_list_file is not None:
                # tell the user to specify the directory list in the directory_list.txt
                text_content = f'There is no directory name given in {self.directory_list_file}.\n'
                text_content += f'To start annotating images, please specify the directory name in {self.directory_list_file} first.'
            else:
                # tell the user to specify the directory list in the directory_list.txt
                text_content = f'There is no directory name given.\n'
                text_content += f'To start annotating images, please specify the directory name.'

            xmin, ymin, dy = 50, 50, 30
            for i, text in enumerate(text_content.split('\n')):
                ymin = ymin + i * dy
                text_width, text_height = cv2.getTextSize(
                    text, self.font, self.directory_name_font_scale, self.line_thickness)[0]

                cv2.rectangle(directory_img, (xmin, ymin), (xmin + text_width + margin, ymin - text_height - margin),
                                self.ACTIVE_BBOX_COLOR, -1)
                cv2.putText(directory_img, text, (xmin, ymin - 5), self.font, 0.6, self.BLACK, self.line_thickness,
                            cv2.LINE_AA)

            cv2.imshow(self.IMAGE_WINDOW_NAME, directory_img)
        else:
            # if we have 24 directories, 5 in each page, then last page index is int(24/5) = 4
            # (i.e., we have 5 pages)
            self.LAST_DIRECTORY_PAGE_INDEX = int(len(self.DIRECTORY_LIST) / self.directory_number_per_page)
            # print('self.LAST_DIRECTORY_PAGE_INDEX =', self.LAST_DIRECTORY_PAGE_INDEX)

            # estimate the directory_name_max_text_width so that we can design how many directorys to show
            # for each row and each column
            directory_name_max_text_width, directory_name_text_height = 0, 0
            # the heights are all same, e.g.,  14, 14, 14, 14, 14, 14, 14, 14,
            # so we do not need self.directory_name_text_heights = []
            for ann_dir in self.DIRECTORY_LIST:
                text_width, directory_name_text_height = cv2.getTextSize(
                    ann_dir, self.font, self.directory_name_font_scale, 1
                    )[0]
                
                if text_width > directory_name_max_text_width:
                    directory_name_max_text_width = text_width

            # set the trackbar for selecting page numbers
            # show the image index bar only if we have more than one class
            checkTrackBarPos = cv2.getTrackbarPos(self.TRACKBAR_DIRECTORY, self.IMAGE_WINDOW_NAME)
            if self.LAST_DIRECTORY_PAGE_INDEX > 1 and checkTrackBarPos == -1:
                cv2.createTrackbar(
                    self.TRACKBAR_DIRECTORY, self.IMAGE_WINDOW_NAME,
                    0, self.LAST_DIRECTORY_PAGE_INDEX,
                    self.set_directory_page_index
                )

            self.directory_list_current_page = []  # initialize the directory list for current page
            if self.directory_page_index <= self.LAST_DIRECTORY_PAGE_INDEX:
                self.directory_list_current_page = self.DIRECTORY_LIST[
                      self.directory_page_index * self.directory_number_per_page:
                      (self.directory_page_index + 1) * self.directory_number_per_page
                ]
            else:
                self.directory_list_current_page = self.DIRECTORY_LIST[
                    self.directory_page_index * self.directory_number_per_page:
                    self.LAST_DIRECTORY_PAGE_INDEX
                ]

            # row and col margin between two bbox in vertical and horizontal
            # directions, respectively.
            row_margin, col_margin = 50, 50
            num_cols = int(self.window_width / (directory_name_max_text_width + margin * 2 + col_margin))
            # print('num_cols:',num_cols)

            # set the window to display at most self.page_max_num_cols columns
            if num_cols > self.page_max_num_cols: 
                num_cols = self.page_max_num_cols
            # print('num_cols:',num_cols)

            col_width = directory_name_max_text_width + margin * 2
            # the height of the rectangle in which the directory name were shown
            row_height = int(directory_name_text_height + margin * 2)

            directory_color = (79, 211, 149)

            self.active_image_annotated_bboxes = []  # initialize the bbox list
            for idx, directory_name in enumerate(self.directory_list_current_page):
                directory_idx = idx + self.directory_page_index * self.directory_number_per_page
                # decide the color of the directory name

                row_id = int(idx / num_cols)  # 0-index
                col_id = int(idx % num_cols)  # 0-index

                xmin = col_margin * (col_id + 1) + col_width * col_id
                ymin = row_margin * (row_id + 1) + row_height * row_id
                xmax, ymax = xmin + col_width, ymin + row_height
                cv2.rectangle(directory_img, (xmin, ymin), (xmax, ymax),
                              directory_color, cv2.FILLED)

                # change the data format not for annotating, but for selecting a specific directory
                # the appended self.directory_exist_list[directory_idx] do not mean a real class index
                bbox = [self.DIRECTORY_LIST[directory_idx], xmin, ymin, xmax, ymax]
                
                # append this bbox to self.image
                self.active_image_annotated_bboxes.append(bbox)

                # text_shadow_color = class_color, text_color = (0, 0, 0), i.e., black
                text_content = directory_name  # str(idx + 1) + '. ' +
                self.draw_directory_name(
                    directory_img, (xmin + margin, ymin + row_height),
                    text_content, directory_color, self.BLACK
                )
    
    def set_directory_page_index(self, ind):
        """
        set the directory page for the object_tracking bar of self.TRACKBAR_DIRECTORY
        the self.directory_page_index value will be set to ind

        This function will only take effect by dragging the trackbar manully
        if we chenge the self.directory_page_index by pressing 'a' or 'd'
        we have to call setTrackbarPos()
        """
        self.directory_page_index = ind
        text = 'Showing directory page {}/{}'.format(
            self.directory_page_index, self.LAST_DIRECTORY_PAGE_INDEX
        )
        self.display_text(text, 3000)

    def draw_directory_name(self, tmp_img, location_to_draw, text_content, text_shadow_color, text_color):
        font = cv2.FONT_HERSHEY_SIMPLEX  # FONT_HERSHEY_SIMPLEX
        font_scale = self.directory_name_font_scale
        margin = 3
        text_width, text_height = cv2.getTextSize(text_content, font, font_scale, self.line_thickness)[0]

        xmin, ymin = location_to_draw[0], location_to_draw[1]
        cv2.rectangle(tmp_img, (xmin, ymin), (xmin + text_width + margin,
                                              ymin - text_height - margin),
                      text_shadow_color, -1)
        cv2.putText(tmp_img, text_content, (xmin, ymin - 5), font, 0.6, text_color, self.line_thickness,
                    cv2.LINE_AA)
      
    def draw_active_directory_bbox(self, tmp_img):
        # self.active_bbox_idx < len(self.active_image_annotated_bboxes) is necessary, otherwise, it cuases error
        # when we changing the image frame in a very fast speed (dragging trackbar) so that setting acitve bbox
        # is not finished
        if self.active_bbox_idx is not None and self.active_bbox_idx < len(self.active_image_annotated_bboxes):
            # do not change the class_index here, otherwise, every time the
            # active_bbox_idx changed (mouse is wandering),
            # the class_index will change (this is not what we want).
            # the data format should be int type, class_idx is 0-index.
            bbox = self.active_image_annotated_bboxes[self.active_bbox_idx]

            _, xmin, ymin, xmax, ymax = bbox
            # we use LINE_THICKNESS = 2
            # line_thickness = 2
            cv2.rectangle(tmp_img, (xmin, ymin), (xmax, ymax),
                          self.ACTIVE_BBOX_COLOR, cv2.FILLED)

            text_content = self.directory_list_current_page[self.active_bbox_idx]
            # text_shadow_color = class_color, text_color = (0, 0, 0), i.e., black
            text_shadow_color = self.ACTIVE_BBOX_COLOR
            text_color = self.BLACK  #
            self.draw_directory_name(tmp_img, (xmin, ymax), text_content, text_shadow_color, text_color)  #

    # def delete_active_directory(self):
    #     if self.active_directory is not None and \
    #             len(self.DIRECTORY_LIST) > 0 and \
    #             self.active_directory in self.DIRECTORY_LIST:
    #         active_directory_index = self.DIRECTORY_LIST.index(self.active_directory )
    #         del self.DIRECTORY_LIST[active_directory_index]

    #         # save the new directory list
    #         with open(self.directory_list_file, 'w') as new_file:
    #             for dir_name in self.DIRECTORY_LIST:
    #                     new_file.write(dir_name + '\n')
    #         new_file.close()

    #     elif self.active_directory is None:
    #         print('The active directory is None')
    #     elif len(self.DIRECTORY_LIST) > 0 and \
    #             self.active_directory not in self.DIRECTORY_LIST:
    #         print('The active directory {} to delete does not exist in
    #         the DIRECTORY_LIST'.format(self.active_directory))

    def update_operating_active_directory_in_txt_file(self):
        '''
        record the operating active directory for the current user so we can conduct batch processing
        using other interface, e.g., MATLAB

            self.active_directory_file is initialized by parent class AnnotationTool based on user name
        Specifying the user name enables different users to use the annotation tool at the same time
        '''

        with open(self.active_directory_file, 'w') as new_file:
            txt_line = self.active_directory
            new_file.write(txt_line + '\n')
        new_file.close()

    # def draw_active_directory_annotated_bbox(self, tmp_img, image_index_str = 0, 
    #                                             image_index_end = None):
    #     if image_index_end == None:
    #         image_index_end = len(self.IMAGE_PATH_LIST) 
    #     for img_path in self.IMAGE_PATH_LIST[image_index_str : image_index_end]:
    #         # add key to the active_directory_annotated_bboxes_dict
            
    #         annotated_bbox = self.active_directory_annotated_bboxes_dict[img_path]
    #         for bbox in annotated_bbox:
    #             self.draw_annotated_bbox_for_clustering(tmp_img,bbox)

    def Event_ConfirmDirectory(self):
        # this is the function called from outside of the this class
        if self.active_bbox_idx is not None and \
            self.directory_list_current_page is not None and \
                self.active_bbox_idx < len(self.directory_list_current_page):
            self.active_directory = self.directory_list_current_page[self.active_bbox_idx]

    def set_active_directory(self):

        self.init_image_window_and_mouse_listener()
        
        while self.active_directory is None:
            # default color (white, [255, 255, 255])
            directory_img = smrc.utils.generate_blank_image(self.window_height, self.window_width)

            if self.directory_list_file is not None:
                self.DIRECTORY_LIST = smrc.utils.load_directory_list_from_file(
                    self.directory_list_file
                )

            # load the directory name for current page for selecting
            # based on the self.DIRECTORY_LIST, self.directory_page_index
            self.load_directory_name_for_current_page(directory_img)

            # EVENT PROCESSING STEP
            self.set_active_bbox_idx_if_NONE()
            
            # set the active directory based on mouse cursor
            self.set_active_bbox_idx_based_on_mouse_position(allow_none=False)
            
            ''' Key Listeners START '''
            pressed_key = self.read_pressed_key()

            # 255 Linux or Windows, cv2.waitKeyEx() & 0xFF , -1
            # Windows cv2.waitKeyEx() or Linux  cv2.waitKey(), 0
            # Windows cv2.waitKey()
            if pressed_key != 255 and pressed_key != 0 and pressed_key != -1:
                print('pressed_key=', pressed_key)  # ('pressed_key=', -1) if no key is pressed.
                # print('pressed_key & 0xFF =', pressed_key & 0xFF)
                # print('self.platform = ', self.platform)
                # handle string key a -z
                if ord('a') <= pressed_key <= ord('z'):  # 'a': 97, 'z': 122
                    if pressed_key == ord('a') or pressed_key == ord('d'):
                        # show previous image key listener
                        if pressed_key == ord('a'):
                            self.directory_page_index = smrc.utils.decrease_index(self.directory_page_index,
                                                                            self.LAST_DIRECTORY_PAGE_INDEX)
                        # show next image key listener
                        elif pressed_key == ord('d'):
                            self.directory_page_index = smrc.utils.increase_index(self.directory_page_index,
                                                                            self.LAST_DIRECTORY_PAGE_INDEX)
                        cv2.setTrackbarPos(self.TRACKBAR_DIRECTORY, self.IMAGE_WINDOW_NAME, self.directory_page_index)

                elif pressed_key & 0xFF == 13:  # Enter key is pressed
                    print('Enter key is pressed.')
                    self.Event_ConfirmDirectory()
                elif pressed_key & 0xFF == 27:  # Esc key is pressed
                    # quit the program
                    print('Esc key is pressed, quit the program.')
                    sys.exit(0)

            if self.game_controller_available and self.game_controller_on:
                # # EVENT PROCESSING STEP
                # self.set_active_bbox_idx_if_NONE()
                
                for event in pygame.event.get():  # User did something
                    if event.type == pygame.QUIT:  # If user clicked close
                        break

                    # Possible joystick actions: JOYAXISMOTION JOYBALLMOTION JOYBUTTONDOWN JOYBUTTONUP JOYHATMOTION
                    if event.type == pygame.JOYBUTTONDOWN:
                        #print("Joystick button pressed.")
                        #print('event =', event)
                        # ('event =', < Event(10-JoyButtonDown {'joy': 0, 'button': 2}) >)
                        # print('event["button"] =', event['button'])  # this is wrong
                        # print('event.button =', event.button) #'event.key =', event.key,

                        # get_button
                        btn_id = event.button  # get the id of the button
                        btn_name = self.button_names[btn_id]  # get the name of the button
                        print("%s pressed" % (btn_name))
                        self.button_states[btn_name] = 1  # set the state of the button to 1

                            # self.set_active_bbox_idx_for_game_controller()
                        if btn_name == 'L1':  # Esc
                            print('L1 pressed')
                            self.quit_annotation_tool = True  # quit the annotation tool
                            print('QUIT key is pressed, quit the program.')
                            sys.exit(0)
                        elif btn_name == 'R1':  # shift key
                            print('R1 pressed')
                            self.Event_ConfirmDirectory()
                            #if self.directory_list_current_page is not None and \
                            #    len(self.directory_list_current_page) > 0:
                            #    self.active_directory = self.directory_list_current_page[0]

            if self.active_bbox_idx is not None:
                # redraw the active directory bbox to make it more visualable
                self.draw_active_directory_bbox(directory_img)

            cv2.imshow(self.IMAGE_WINDOW_NAME, directory_img)

            if self.WITH_QT:
                # if window gets closed then quit
                if cv2.getWindowProperty(self.IMAGE_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    print('Window is closed, quit the program.')
                    sys.exit(0)

        # close the window
        cv2.destroyWindow(self.IMAGE_WINDOW_NAME)

        print(self.active_directory, '  selected')

        return self.active_directory 
