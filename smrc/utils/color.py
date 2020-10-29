import numpy as np
import random


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
ACTIVE_BBOX_COLOR = (0, 255, 255)  # yellow, the RGB of yellow is (255, 255,0 ), but here we use BGR format
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
# UnKnown = (255, 255, 0)
Orange = (0, 128, 255)
Pink = (255, 0, 255)
#
# color = tuple([random.randint(0, 255) for _ in range(3)])

color_map = [np.array([255, 0, 0]),
                   np.array([0, 255, 0]),
                   np.array([0, 0, 255]),
                   np.array([125, 125, 0]),
                   np.array([0, 125, 125]),
                   np.array([125, 0, 125]),
                   np.array([50, 100, 50]),
                   np.array([100, 50, 100])]

# to extend the color in the future.
#     (0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 255, 255),
#     (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0),
#     (128, 128, 0), (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128)
class_bgr = [
    (79, 211, 149), (255, 132, 132), (70, 79, 158), (193, 255, 0), (149, 132, 0),
    (220, 158, 246), (255, 18, 211), (106, 26, 123), (97, 18, 246)]

# generate random colors

# from mask r-cnn

# the final definitions of the colors of the classes for bbox plotting
# make sure we have defined enough class colors for the loaded class names.
CLASS_BGR_COLORS = np.array(class_bgr)


import colorsys
# from deep sort yolo.py


def unique_colors(num_colors, random_seed=None):
    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / num_colors, 1., 1.)
                  for x in range(num_colors)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    if random_seed is None:
        random_seed = 10101
    random.seed(random_seed)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to de-correlate adjacent classes.
    random.seed(None)  # Reset seed to default.

    return colors
