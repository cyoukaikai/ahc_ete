import pygame
import sys
# import pygame.locals
import cv2
pygame.init()

screen = pygame.display.set_mode((300,200))

running = True


while running:
    # tmp_image = cv2.imread('tests.jpg')
    # cv2.imshow('tests', tmp_image)
    for event in pygame.event.get():  # User did something
        if event.type == pygame.QUIT:  # If user clicked close
            break
        print(f'event = {event}, event.type={event.type}')
        # sys.exit(0)
        ''' Key Listeners START '''
        if event.type == pygame.KEYDOWN or event.type == pygame.KEYUP:
            # self.keyboard_listener(event, tmp_img)
            print('Enter key listener')
        else:
            print('Enter gamer controller listener')
            # self.game_controller_listener(event, tmp_img)
    #
# while running:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False
#         elif event.type == pygame.KEYDOWN:
#             # return a tuple of a large number of boolean values, while only the position ofthe pressed key
#             # has 1 value.
#             keys = pygame.key.get_pressed()
#             print(f'{keys} pressed...')
#             print(f'{event.key} pressed...')
#             print(f'{pygame.key.name(event.key)} pressed...')
#             # self.assertEqual(pygame.key.name(pygame.K_RETURN), "return")
#             # self.assertEqual(pygame.key.name(pygame.K_0), "0")
#             # #
#             if event.key == pygame.K_ESCAPE:
#                 running = False
#             elif event.key == pygame.K_a and pygame.key.get_mods() & pygame.KMOD_SHIFT:
#                 print("pressed: SHIFT + A")
#             elif event.key == pygame.K_a:
#                 print("pressed: SHIFT + A")
# pygame.quit()


# (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) pressed...
