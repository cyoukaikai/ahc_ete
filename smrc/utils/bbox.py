import numpy as np
import os
from .base import get_file_list_in_directory, get_dir_list_in_directory


# code from https://www.life2coding.com/convert-image-frames-video-file-using-opencv-python/
def get_bbox_area(x1, y1, x2, y2):
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    return width * height


def bbox_area(bbox):
    x1, y1, x2, y2 = bbox[1:5]
    return get_bbox_area(x1, y1, x2, y2)


def get_bbox_rect(bbox):
    return bbox[1:5]


def get_bbox_class(bbox):
    return bbox[0]


def compute_iou(a, b, epsilon=1e-5):
    """ Given two boxes `a` and `b` defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union score for these two boxes.

    Args:
        a:          (list of 4 numbers) [x1,y1,x2,y2]
        b:          (list of 4 numbers) [x1,y1,x2,y2]
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (float) The Intersect of Union score.
    """
    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width < 0) or (height < 0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined+epsilon)
    return iou


# def my_compute_iou(box1_rect, box2_rect):
#     """My implement the intersection over union (IoU) between box1_rect and box2_rect
#     box1_rect: -- first box, list object with coordinates (x1, y1, x2, y2)
#     box2_rect: -- second box, list object with coordinates (x1, y1, x2, y2)
#     :param box1_rect: rectangle not the bbox we used
#     :param box2_rect: rectangle not the bbox we used:
#     :return:
#     """
#
#     #  Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1_rect and box2_rect. Calculate its Area.
#     # print('box1_rect =', box1_rect)
#     # print('box2_rect =', box2_rect)
#     xi1 = max(box1_rect[0], box2_rect[0])
#     yi1 = max(box1_rect[1], box2_rect[1])
#     xi2 = min(box1_rect[2], box2_rect[2])
#     yi2 = min(box1_rect[3], box2_rect[3])
#
#     # intersection area
#     inter_area = max(xi2 - xi1 + 1, 0) * max(yi2 - yi1 + 1, 0)
#     # inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
#
#     # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
#     # box1_rect_area = (box1_rect[2] - box1_rect[0]) * (box1_rect[3] - box1_rect[1])
#     # box2_rect_area = (box2_rect[2] - box2_rect[0]) * (box2_rect[3] - box2_rect[1])
#     box1_rect_area = (box1_rect[2] - box1_rect[0] + 1) * (box1_rect[3] - box1_rect[1] + 1)
#     box2_rect_area = (box2_rect[2] - box2_rect[0] + 1) * (box2_rect[3] - box2_rect[1] + 1)
#
#     union_area = float(box1_rect_area + box2_rect_area - inter_area)
#
#     # compute the IoU
#     iou = inter_area / union_area
#     # print('inter_area =', inter_area, 'union_area = ', union_area, 'iou = ', iou)
#     return iou


def batch_iou_vec(a, b, epsilon=1e-5):
    """ Given two arrays `a` and `b` where each row contains a bounding
        box defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union scores for each corresponding
        pair of boxes.

    Args:
        a:          (numpy array) each row containing [x1,y1,x2,y2] coordinates
        b:          (numpy array) each row containing [x1,y1,x2,y2] coordinates
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (numpy array) The Intersect of Union scores for each pair of bounding
        boxes.
    """
    # COORDINATES OF THE INTERSECTION BOXES
    assert a.shape == b.shape, \
        f'Given pairs of bbox should have the same shape, but' \
        f' ({a.shape}, {b.shape})..'

    x1 = np.array([a[:, 0], b[:, 0]]).max(axis=0)
    y1 = np.array([a[:, 1], b[:, 1]]).max(axis=0)
    x2 = np.array([a[:, 2], b[:, 2]]).min(axis=0)
    y2 = np.array([a[:, 3], b[:, 3]]).min(axis=0)

    # AREAS OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)

    # handle case where there is NO overlap
    width[width < 0] = 0
    height[height < 0] = 0

    area_overlap = width * height

    # COMBINED AREAS
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined + epsilon)
    return iou


def load_bbox_from_file(ann_path):
    annotated_bbox = []
    if os.path.isfile(ann_path):
        with open(ann_path, 'r') as old_file:
            lines = old_file.readlines()
        old_file.close()

        # print('lines = ',lines)
        for line in lines:
            result = line.rstrip('\n').split(' ')

            # the data format in line (or txt file) should be int type, 0-index.
            # we transfer them to int again even they are already in int format (just in case they are not)
            bbox = [int(result[0]), int(result[1]), int(result[2]), int(
                result[3]), int(result[4])]
            annotated_bbox.append(bbox)
    return annotated_bbox


def bbox_transfer_to_txt_line_format(class_idx, xmin, ymin, xmax, ymax):
    items = map(str, [class_idx, xmin, ymin, xmax, ymax])
    return ' '.join(items)


def save_bbox_to_file(ann_path, bbox_list):
    # save the bbox if it is not the active bbox (future version, active_bbox_idxs includes more than one idx)
    with open(ann_path, 'w') as new_file:
        for bbox in bbox_list:
            class_idx, xmin, ymin, xmax, ymax = bbox
            txt_line = bbox_transfer_to_txt_line_format(
                int(class_idx), round(xmin), round(ymin), round(xmax), round(ymax)
            )
            # we need to add '\n'(newline), otherwise, all the bboxes will be in one line and not able to be recognized.
            new_file.write(txt_line + '\n')
    new_file.close()


def save_bbox_to_file_incrementally(ann_path, bbox_list):
    old_bbox_list = load_bbox_from_file(ann_path=ann_path)

    # save the bbox if it is not the active bbox (future version, active_bbox_idxs includes more than one idx)
    with open(ann_path, 'a') as new_file:
        for bbox in bbox_list:
            # if this bbox is already in the file, then not add it.
            if bbox in old_bbox_list:
                continue

            class_idx, xmin, ymin, xmax, ymax = bbox
            txt_line = bbox_transfer_to_txt_line_format(class_idx, xmin, ymin, xmax, ymax)
            # we need to add '\n'(newline), otherwise, all the bboxes will be in one line and not able to be recognized.
            new_file.write(txt_line + '\n')

            # update the old_bbox_list so that even there is redundant bbox in
            # the bbox_list, only one of them will be added.
            old_bbox_list.append(bbox)
    new_file.close()


def save_bbox_to_file_overlap_delete_former(ann_path, bbox_list, overlap_iou_thd):
    bbox_list_old = load_bbox_from_file(ann_path=ann_path)
    bbox_list_all = bbox_list_old + bbox_list

    for bbox in bbox_list:
        for bbox_old in bbox_list_old:
            if compute_iou(bbox[1:5], bbox_old[1:5]) > overlap_iou_thd:
                bbox_list_all.remove(bbox_old)
        bbox_list_old.append(bbox)
    save_bbox_to_file(ann_path=ann_path, bbox_list=bbox_list_all)


def save_bbox_to_file_overlap_delete_latter(ann_path, bbox_list, overlap_iou_thd):
    bbox_list_all = load_bbox_from_file(ann_path=ann_path)
    for bbox in bbox_list:
        overlap_flag = False
        for bbox_old in bbox_list_all:
            if compute_iou(bbox[1:5], bbox_old[1:5]) > overlap_iou_thd:
                overlap_flag = True
                break
        if not overlap_flag:
            bbox_list_all.append(bbox)

    save_bbox_to_file(ann_path=ann_path, bbox_list=bbox_list_all)


def is_valid_bbox_rect(xmin, ymin, xmax, ymax, image_width, image_height):
    if 0 <= min(xmin, xmax) < image_width and \
            0 <= max(xmin, xmax) < image_width and \
            0 <= min(ymin, ymax) < image_height and \
            0 <= max(ymin, ymax) < image_height and \
            xmin < xmax and ymin < ymax:
        return True
    else:
        return False


def post_process_bbox_margin_check(bbox, image_width, image_height):
    class_index, xmin, ymin, xmax, ymax = bbox
    xmin, ymin, xmax, ymax = post_process_bbox_coordinate(xmin, ymin, xmax, ymax, image_width, image_height)
    return [class_index, xmin, ymin, xmax, ymax]


def post_process_bbox_coordinate(xmin, ymin, xmax, ymax, image_width, image_height):
    # if the bbox is valid, return it back
    # save the coordinates before modification
    x1, y1, x2, y2 = xmin, ymin, xmax, ymax

    modified = False
    if xmin < 0 or xmax < 0 or ymin < 0 or ymax < 0:
        if xmin < 0:
            xmin = 0
        if xmax < 0:
            xmax = 0
        if ymin < 0:
            ymin = 0
        if ymax < 0:
            ymax = 0
        modified = True
    if xmin >= image_width or xmax >= image_width or \
            ymin >= image_height or ymax >= image_height:

        if xmin >= image_width:
            xmin = image_width - 1

        if xmax >= image_width:
            xmax = image_width - 1

        if ymin >= image_height:
            ymin = image_height - 1

        if ymax >= image_height:
            ymax = image_height - 1
        modified = True
    if xmin >= xmax or ymin >= ymax:
        if xmin > xmax:
            xmin, xmax = min(xmin, xmax), max(xmin, xmax)
        elif xmin == xmax:
            if xmin + 5 < image_width:
                xmax = xmin + 5
            else:
                xmin = xmax - 5

        if ymin > ymax:
            ymin, ymax = min(ymin, ymax), max(ymin, ymax)
        elif ymin == ymax:
            if ymin + 5 < image_height:
                ymax = ymin + 5
            else:
                ymin = ymax - 5
        modified = True
    if modified:
        print('=====================================================')
        print('before: x1 = {}, y1 = {}, x2 = {}, y2 = {}'.format(
            x1, y1, x2, y2)
        )
        print('after: x1 = {}, y1 = {}, x2 = {}, y2 = {}'.format(
            xmin, ymin, xmax, ymax)
        )
        print('------------------------------------------------------')
    return [round(xmin), round(ymin), round(xmax), round(ymax)]


def post_process_bbox_list(bbox_list, image_height, image_width, class_list=None):
    bbox_processed = []
    for bbox_idx, bbox in enumerate(bbox_list):
        class_idx, xmin, ymin, xmax, ymax = bbox

        if class_list is not None and class_idx not in class_list:
            print(f'The class {class_idx} for [{bbox}] is not in class_list {class_list}, deleted ...')
            continue
        elif xmin == xmax or ymin == ymax:
            print(f'xmin = {xmin}, xmax = {xmax}, ymin = {ymin}, ymax = {ymax}, deleted ...')
            continue
        else:
            if not is_valid_bbox_rect(xmin, ymin, xmax, ymax, image_width, image_height):
                print(f'{bbox} is invalid, image_height = {image_height}, image_width = {image_width}')
                xmin, ymin, xmax, ymax = post_process_bbox_coordinate(
                    xmin, ymin, xmax, ymax,
                    image_width,
                    image_height
                )
                bbox = [class_idx, xmin, ymin, xmax, ymax]
            bbox_processed.append(bbox)
    return bbox_processed


def empty_annotation_file_list(ann_path_list):
    print(f'Deleting annotation for {len(ann_path_list)} files.')
    for ann_path in ann_path_list:
        print(f'Deleting annotation for {ann_path}...')
        empty_annotation_file(ann_path)


def empty_annotation_file(ann_path):
    try:
        open(ann_path, 'w').close()
    except FileNotFoundError:
        print(f'ann_path = {ann_path} does not exist, can not be cleaned.')


def delete_any_specified_bbox_coordinate(ann_path, bbox_list, bbox):
    try:
        bbox_idx = bbox_list.index(bbox)
    except IndexError:
        print(f'{bbox} not found in {bbox_list} for {ann_path}, please check ...')

    del bbox_list[bbox_idx]

    save_bbox_to_file(ann_path=ann_path, bbox_list=bbox_list)


def replace_one_bbox(ann_path, bbox_old, bbox_new):
    bbox_list = load_bbox_from_file(ann_path)
    if bbox_old in bbox_list:
        bbox_idx = bbox_list.index(bbox_old)
        bbox_list[bbox_idx] = bbox_new
        save_bbox_to_file(ann_path, bbox_list)
    else:
        print(f'{bbox_old} not found in {bbox_list} for {ann_path}, please check ...')


def delete_any_specified_bbox_idx(ann_path, bbox_list, bbox_idx):
    try:
        del bbox_list[bbox_idx]
    except IndexError:
        print(f'Error when deleting bbox idx {bbox_idx} from {bbox_list}')
    save_bbox_to_file(ann_path=ann_path, bbox_list=bbox_list)


def delete_one_bbox_from_file(ann_path, bbox):
    bbox_list = load_bbox_from_file(ann_path)
    if bbox in bbox_list:
        delete_any_specified_bbox_coordinate(ann_path=ann_path, bbox_list=bbox_list, bbox=bbox)
    else:
        print(f'{bbox} not found in {bbox_list} for {ann_path}, please check ...')

# def exchange_image_dir_and_ann_dir(dir_name_to_process, source_dir, target_dir):
#     """
#     instead of directly replace(source_dir, target_dir, 1), which may be
#     not accurate, e.g., images/1/0001.jpg -> 1/1/0001.txt
#     :param dir_name_to_process:
#     :param source_dir:
#     :param target_dir:
#     :return:
#     """
#     source_dir_str = os.path.sep + source_dir + os.path.sep
#     target_dir_str = os.path.sep + target_dir + os.path.sep
#     result_dir = dir_name_to_process.replace(source_dir_str, target_dir_str, 1)
#     return result_dir


def transfer_bbox_list_xywh_format(bbox_list, with_class_index=True):
    transferred_bbox_list = []
    for bbox in bbox_list:
        transferred_bbox_list.append(
            bbox_to_xywh(bbox, with_class_index)
        )
    return transferred_bbox_list


def bbox_to_xywh(bbox, with_class_index=True):
    x = (bbox[1] + bbox[3]) / 2.0
    y = (bbox[2] + bbox[4]) / 2.0
    w = bbox[3] - bbox[1]
    h = bbox[4] - bbox[2]

    if with_class_index:
        return [bbox[0], x, y, w, h]
    else:
        return [x, y, w, h]


def get_bbox_rect_center(bbox):
    xo = (bbox[1] + bbox[3]) / 2.0
    yo = (bbox[2] + bbox[4]) / 2.0
    return [xo, yo]


def bbox_to_yolo_format(bbox, image_width, image_height, with_class_index=True):
    # borrowed from OpenLabeling
    # YOLO wants everything normalized
    # print(point_1, point_2, image_width, image_height)
    # Order: class x_center y_center x_width y_height
    class_index, x1, y1, x2, y2 = bbox
    x = (x1 + x2) / (2.0 * image_width)
    y = (y1 + y2) / (2.0 * image_height)
    w = float(abs(x2 - x1)) / image_width
    h = float(abs(y2 - y1)) / image_height
    if with_class_index:
        return [class_index, x, y, w, h]
    else:
        return [x, y, w, h]


def bbox_to_tlwh(bbox, with_class_index=True):
    class_idx, x1, y1 = bbox[:3]
    w = bbox[3] - bbox[1]
    h = bbox[4] - bbox[2]
    if with_class_index:
        return [class_idx, x1, y1, w, h]
    else:
        return [x1, y1, w, h]


def transfer_bbox_list_x1y1wh_format(bbox_list, with_class_index=True):
    transferred_bbox_list = []
    for bbox in bbox_list:
        transferred_bbox_list.append(
            bbox_to_tlwh(bbox, with_class_index)
        )
    return transferred_bbox_list


def bbox_to_xyah(bbox, with_class_index=True):
    """Convert bounding box to format `(center x, center y, aspect ratio,
    height)`, where the aspect ratio is `width / height`.
    from deep_sort/object_detection.py
    def to_xyah(self):
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
    """
    x = (bbox[1] + bbox[3]) / 2.0
    y = (bbox[2] + bbox[4]) / 2.0
    w = bbox[3] - bbox[1]
    h = bbox[4] - bbox[2]
    a = w / h
    if with_class_index:
        return [bbox[0], x, y, a, h]
    else:
        return [x, y, a, h]




