import pygame
import cv2
import os
import random
from .music import Music
import smrc.utils

pygame.init()

# Initialize the joysticks
pygame.joystick.init()


# root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'media')
# # /home/kai/optimize_tool/smrc/media
# # print(f'root_dir = {root_dir}')
# print(f'Loading music playlist from {root_dir} ..')
# playlist = smrc.line.get_file_list_recursively(root_dir, ext_str='.mp3')
# print(f'self.music_playlist = {playlist}')

# def init_music_playlist():
# init_music_playlist()


class GameController(Music):
    def __init__(self, music_on=False):
        super().__init__(music_on)

        self.game_controller_available = False
        self.game_controller_on = True
        self.joystick = None

        # We'll store the states here.
        self.axis_states = {}
        self.button_states = {}
        self.hat_states = {}

        # These constants were borrowed from linux/input.h
        self.axis_names = {}
        self.button_names = {}
        self.hat_names = {}

        self.axis_map = []
        self.button_map = []
        self.hat_map = []

        self.game_controller_axis_moving_on = False
        self.game_controller_axis1_moving_on = False

        # self.music_playlist = []
        # # code for pygame joystick from https://www.pygame.org/docs/ref/joystick.html
        # pygame.init()
        # # Used to manage how fast the screen updates
        # # self.clock = pygame.time.Clock()
        #
        # # Initialize the joysticks
        # pygame.joystick.init()

        joystick_count = pygame.joystick.get_count()
        print("Number of joysticks: {}".format(joystick_count))

        # self.init_music_playlist()

        if joystick_count > 0:
            # we always use first game controller even if more than one are detected.
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()

            # Get the name from the OS for the controller/joystick
            self.js_name = self.joystick.get_name()
            print('Device name: %s' % self.js_name)
            # IFYOO Game for Windows
            
            self.hat_names = {
                0: 'hat0'  # value format (0, 1)
                #1: 'hat1'
            }

            # use test_button_axis_hat() function to assit the defination of the names of the buttons,
            # axis, and hat.

            # 'Elecom JC-U3613M' in linux, and 'Controller (JC-U3613M - Xinput Mode)' in Windows
            if self.js_name.find('JC-U3613M') > 0:
                # note that axises 2 and 5 are in pair, 3 and 4 are in pair.
                self.axis_names = {
                    0: 'x',
                    1: 'y',
                    2: 'tx',  #once pressed, always -1.00 
                    3: 'ry',
                    4: 'rx',
                    5: 'ty'  # once pressed, always -1.00,
                }
                self.button_names = {
                    0: 'A',
                    1: 'B',
                    2: 'X',
                    3: 'Y',
                    4: 'L1',
                    5: 'R1',
                    6: 'back',
                    7: 'start',
                    8: 'mode'
                    #9, 10  not defined in the game controller (can not those buttons)
                }
                
            else: #self.js_name == 'SHANWAN IFYOO Gamepad': #SHANWAN IFYOO Gamepad
                self.axis_names = {
                    0: 'x',
                    1: 'y',
                    2: 'rx',
                    3: 'ry'
                }
                #try not to use buttons {'L2', 'R2','select'}, as they maybe not defined for different game controller
                self.button_names = {
                    0: 'Y',
                    1: 'B',
                    2: 'A',
                    3: 'X',
                    4: 'L1',
                    5: 'R1',
                    6: 'L2', # try not to use this button
                    7: 'R2', # try not to use this button
                    8: 'select',  # try not to use this button
                    9: 'start',
                    12: 'mode'  #SHANWAN IFYOO Gamepad 
                    #button 10 and button 12 will not exist at the same time, so do not worry the name issue
                }
            num_buttons = self.joystick.get_numbuttons()
            print("Number of buttons: {}".format(num_buttons))

            for i in range(num_buttons):
                btn_name = self.button_names.get(i, 'unknown(0x%01x)' % i)
                self.button_map.append(btn_name)
                self.button_states[btn_name] = 0
                print(' button {}, name {} '.format(i, btn_name) )

            # Usually axis run in pairs, up/down for one, and left/right for
            # the other.
            num_axes = self.joystick.get_numaxes()
            print("Number of axes: {}".format(num_axes))

            for i in range(num_axes):
                axis_name = self.axis_names.get(i, 'unknown(0x%02x)' % i)
                self.axis_map.append(axis_name)
                self.axis_states[axis_name] = 0.0
                print(' axis {}, name {} '.format(i, axis_name) )
            num_hats = self.joystick.get_numhats()
            print("Number of hats: {}".format(num_hats))

            for i in range(num_hats):
                hat_name = self.hat_names.get(i, 'unknown(0x%03x)' % i)
                self.hat_map.append(hat_name)
                self.hat_states[hat_name] = (0, 0) #0.0
                print(' hat {}, name {} '.format(i, hat_name) )

            print('You can use smrc/show_joystick_map.py to display the button, axis and hat ID.')
            print('Also check which button, axis, hat is pressed to understand the reference.')
            # if we find any joystick
            self.game_controller_available = True
            print('Game controller is ready for use.')

    # def init_music_playlist(self):
    #     root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'media')
    #     # /home/kai/optimize_tool/smrc/media
    #     # print(f'root_dir = {root_dir}')
    #     print(f'Loading music playlist from {root_dir} ..')
    #     self.music_playlist = smrc.line.get_file_list_recursively(root_dir, ext_str='.mp3')
    #     print(f'self.music_playlist = {self.music_playlist}')
    #
    #     pygame.mixer.music.load(self.music_playlist.pop())
    #     # Get the first object_tracking from the playlist
    #     # pygame.mixer.music.queue(self.music_playlist.pop())  # Queue the 2nd song
    def is_axis_triggered(self, axis_name, axis_value):
        assert axis_name in self.axis_states

        if abs(axis_value) > abs(self.axis_states[axis_name]) and \
                abs(axis_value) > 0.2:
            return True
        else:
            return False

    def test_button_axis_hat(self):
        '''
        usage: 
            from smrc.game_controller import GameController 
            game_controller = GameController()
            game_controller.test_button_axis_hat()

        '''
        if not self.game_controller_available:
            print('No game controller is available.')
        else:
            print('Testing game controller ...')
            print('Press a button, axis, or hat to see its name ...')
            while True:
                # EVENT PROCESSING STEP
                for event in pygame.event.get():  # User did something
                    if event.type == pygame.QUIT:  # If user clicked close
                        break

                    # Possible joystick actions: JOYAXISMOTION JOYBALLMOTION JOYBUTTONDOWN JOYBUTTONUP JOYHATMOTION
                    if event.type == pygame.JOYBUTTONDOWN or event.type == pygame.JOYBUTTONUP:
                        print('event =', event)
                        # ('event =', < Event(10-JoyButtonDown {'joy': 0, 'button': 2}) >)
                        
                        # print('event["button"] =', event['button'])  # this is wrong
                        # print('event.button =', event.button) #'event.key =', event.key,

                        # get_button ID and button name
                        btn_id = event.button  # get the id of the button
                        btn_name = self.button_names[btn_id]  # get the name of the button
                        if event.type == pygame.JOYBUTTONDOWN:
                            #print("Joystick button pressed.")
                            print("%s pressed" % (btn_name))
                        elif event.type == pygame.JOYBUTTONUP:
                            #print("Joystick button released.")
                            print("%s released" % (btn_name))
            
                    # JOYAXISMOTION parameter:  joy, hat, value
                    if event.type == pygame.JOYHATMOTION:
                        #print("Joystick hat pressed.")
                        #print('event =', event)

                        hat_id, hat_value = event.hat, event.value  # get the id of the hat
                        hat_name = self.hat_names[hat_id]  # get the name of the hat
                        print("%s pressed, " % (hat_name))
                        print("hat value  : {}".format(hat_value))
                        

                    # JOYAXISMOTION parameter:  joy, axis, value
                    if event.type == pygame.JOYAXISMOTION:
                        #print("Joystick axis pressed.")
                        print('event =', event)
                        # ('event =', < Event(7-JoyAxisMotion {'joy': 0, 'value': 0.0, 'axis': 3}) >)

                        # get_axis
                        axis_id = event.axis  # get the id of the axis
                        axis_name = self.axis_names[axis_id]  # get the name of the axis
                        print("%s axis pressed" % (axis_name))
                        axis_value = event.value
                        print("axis value  : {}".format(axis_value))
                
                pressed_key = cv2.waitKey(20)
                if pressed_key & 0xFF == 27:  # Esc key is pressed
                    # cv2.destroyWindow(self.IMAGE_WINDOW_NAME)
                    break

    def Event_MoveToNextMusic(self):
        # print(f'self.music_playlist = {self.music_playlist}')
        if self.play_music_on and len(self.music_playlist) > 0:
            next_song = random.choice(self.music_playlist)
            print(f'Move to next music {next_song} ...')
            pygame.mixer.music.load(next_song)
            if self.game_controller_available:
                pygame.mixer.music.play()  # Play the music
            else:
                pygame.mixer.music.play(-1)  # Play the music

        # if len(playlist) > 0:
        #     pygame.mixer.music.queue(playlist.pop())  # Queue the next one in the list

    def init_and_play_music(self):
        if self.play_music_on:
            # print(f'self.play_music_on = {self.play_music_on}')
            print('Music mode is turned on, to load and play music now ...')
            self.init_music_playlist()

            if len(self.music_playlist) > 0:
                pygame.mixer.music.load(self.music_playlist[0])
                print(f'Playing music {self.music_playlist[0]}')
                # Get the first object_tracking from the playlist
                # pygame.mixer.music.queue(self.music_playlist.pop())  # Queue the 2nd song

                if self.game_controller_available:
                    # pygame.mixer.music.set_endevent(pygame.USEREVENT)  # Setup the end object_tracking event
                    pygame.mixer.music.play()  # Play the music
                else:
                    pygame.mixer.music.play(-1)  # Play the music
