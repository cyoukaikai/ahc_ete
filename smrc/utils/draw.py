import cv2
import numpy as np
from .color import *


def put_text_on_image(
        tmp_img, text_content, x0=20, y0=60, dy=50,
        font_color=(0, 0, 255), thickness=2, font_scale=0.6
        ):
    for i, text in enumerate(text_content.split('\n')):
        y = y0 + i * dy
        cv2.putText(tmp_img, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color=font_color, thickness=thickness)


def display_text_on_image(tmp_img, text_content, font_scale=0.8):
    x, y0, dy = 20, 60, 50
    for i, text in enumerate(text_content.split('\n')):
        y = y0 + i * dy
        cv2.putText(tmp_img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color=(0, 0, 255), thickness=3)


def display_text_on_image_top_middle(tmp_img, text_content, font_color=None):
    if font_color is None:
        font_color = RED

    # defined in put_text_on_image
    font = cv2.FONT_HERSHEY_SIMPLEX

    # change the font scale if the image window is of small size
    height, width = tmp_img.shape[:2]

    if width < 1000:
        font_scale = 0.6
        y, dy = 40, 40
        line_thickness = 2
    else:
        font_scale = 1
        y, dy = 50, 50  # y vertical margin from the top, and line height
        line_thickness = 3
    # window_width = 1000
    max_text_width = 0
    for i, text in enumerate(text_content.split('\n')):
        text_width, text_height = cv2.getTextSize(text, font, font_scale, line_thickness)[0]
        if text_width > max_text_width:
            max_text_width = text_width
    # print(f'window_width={window_width}')
    # print(f'window_height={window_height}')
    # print(f'font_scale={font_scale}')
    # print(f'text_content={text_content}')
    # print(f'max_text_width = {max_text_width}')
    # sys.exit(0)

    # note that it is not the (window_width - max_text_width/2.0
    # but (tmp_image width - max_text_width)/2.0
    x = int((width - max_text_width)/2.0)
    # print(f'x={x}')
    put_text_on_image(tmp_img, text_content, x, y, dy, font_color, font_scale=font_scale, thickness=line_thickness)


def draw_corners_on_image(tmp_img, corners, radius=3, point_color=(0, 0, 0)):
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(tmp_img, (x, y), radius, point_color)


def draw_edges(tmp_img):
    blur = cv2.bilateralFilter(tmp_img, 3, 75, 75)
    edges = cv2.Canny(blur, 150, 250, 3)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    # Overlap image and edges together
    tmp_img = np.bitwise_or(tmp_img, edges)
    # tmp_img = cv2.addWeighted(tmp_img, 1 - edges_val, edges, edges_val, 0)
    return tmp_img  # have to return tmp_img


def draw_arrow(tmp_image, p, q, color=(255, 255, 255), arrow_magnitude=9, thickness=1, line_type=8, shift=0):
    # adapted from http://mlikihazar.blogspot.com.au/2013/02/draw-arrow-opencv.html
    # draw arrow tail
    cv2.line(tmp_image, p, q, color, thickness, line_type, shift)
    # calc angle of the arrow
    angle = np.arctan2(p[1] - q[1], p[0] - q[0])
    # starting point of first line of arrow head
    pt1 = (int(q[0] + arrow_magnitude * np.cos(angle + np.pi / 4)),
           int(q[1] + arrow_magnitude * np.sin(angle + np.pi / 4)))
    # draw first half of arrow head
    cv2.line(tmp_image, pt1, q, color, thickness, line_type, shift)

    # starting point of second line of arrow head
    pt2 = (int(q[0] + arrow_magnitude * np.cos(angle - np.pi / 4)),
           int(q[1] + arrow_magnitude * np.sin(angle - np.pi / 4)))
    # draw second half of arrow head
    cv2.line(tmp_image, pt2, q, color, thickness, line_type, shift)
    # cv2.line(image, pt1, pt2, (255, 255, 255), thickness, line_type, shift)


def draw_bbox_legend(
        tmp_img, text_content, location_to_draw,
        text_shadow_color, text_color,
        font_scale=0.6, line_thickness=2
):
    font = cv2.FONT_HERSHEY_SIMPLEX  # FONT_HERSHEY_SIMPLEX
    margin = 3
    text_width, text_height = cv2.getTextSize(text_content, font, font_scale, line_thickness)[0]

    xmin, ymin = location_to_draw[0], location_to_draw[1]
    cv2.rectangle(
        tmp_img, (xmin, ymin),
        (xmin + text_width + margin, ymin - text_height - margin),
        text_shadow_color, -1
    )
    cv2.putText(
        tmp_img, text_content, (xmin, ymin - 5),
        font, 0.6, text_color, line_thickness,
        cv2.LINE_AA
    )