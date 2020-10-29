import json
import imagesize
import os
from ..base import *
from .detection_process import nms_detection_dict, det_dict_to_smrc_tracking_format, \
    image_path_last_two_level, filter_out_low_score_detection_dict


def yolo_bbox_rect_to_smrc_rect(bbox_rect, img_h, img_w):
    center_x, center_y, width, height = bbox_rect
    x1 = round((center_x - 0.5 * width) * img_w)
    x2 = round((center_x + 0.5 * width) * img_w)
    y1 = round((center_y - 0.5 * height) * img_h)
    y2 = round((center_y + 0.5 * height) * img_h)
    if x1 < 0: x1 = 0
    if x2 > img_w: x2 = img_w - 1
    if y1 < 0: y1 = 0
    if y2 > img_h: y2 = img_h - 1
    return [x1, y1, x2, y2]


def parse_yolov4_frame_det(frame_det):
    dets = frame_det['objects']
    if len(dets) == 0:
        return

    #  "filename":"test_data/test_images/2/0003.jpg",
    image_path = frame_det['filename']
    assert os.path.isfile(image_path)
    # tmp_img = cv2.imread(image_path)
    img_w, img_h = imagesize.get(image_path)

    det_bbox_list = []
    for obj in dets:
        class_idx = obj['class_id']
        yolo_bbox_rect = obj['relative_coordinates']
        score = obj['confidence']
        bbox_rect = [
            yolo_bbox_rect['center_x'],
            yolo_bbox_rect['center_y'],
            yolo_bbox_rect['width'],
            yolo_bbox_rect['height'],
        ]
        x1, y1, x2, y2 = yolo_bbox_rect_to_smrc_rect(bbox_rect, img_h, img_w)
        det_bbox_list.append([class_idx, x1, y1, x2, y2, score])
    frame_det_dict = {
        "image_path": image_path,
        "det_bbox_list": det_bbox_list,
    }
    return frame_det_dict


def json_yolov4_to_yolov3(json_yolov4_dir, json_yolov3_dir):
    generate_dir_if_not_exist(json_yolov3_dir)
    json_file_list = get_file_list_in_directory(
        json_yolov4_dir, ext_str='.json', only_local_name=True
    )
    for k, json_file in enumerate(json_file_list):
        output_json_file = os.path.join(json_yolov3_dir, json_file)
        input_json_file = os.path.join(json_yolov4_dir, json_file)

        det_list_v3 = []
        with open(input_json_file, 'rb') as input_json:
            det_list_v4 = json.load(input_json)

        for frame_det in det_list_v4:
            # skip the empty detection
            if len(frame_det['objects']) == 0:
                continue

            frame_det_dict = parse_yolov4_frame_det(frame_det)
            image_path = frame_det_dict["image_path"]
            for obj in frame_det_dict["det_bbox_list"]:
                class_idx, x1, y1, x2, y2, score = obj
                det_list_v3.append(
                    {"image_path": image_path, "category_id": class_idx, "bbox": [x1, y1, x2, y2], "score": score}
                )

        with open(output_json_file, 'w') as fp:
            fp.write(
                '[\n' +
                ',\n'.join(json.dumps(one_det) for one_det in det_list_v3) +
                '\n]')
        # with open(output_json_file, 'w') as list_v3:
        #     # json.dump dumps all its content in one line.
        #     # indent=2 is to record each dictionary entry on a new line
        #     json.dump(det_list_v3, list_v3, sort_keys=True, indent=2, separators=(',', ':'))
        print(f'Processing {input_json_file} to {output_json_file} done [{k+1}/{len(json_file_list)}]... ')


def load_yolov4_json_detection_to_dict(
        json_detection_file, detection_dict=None,
        score_thd=None, nms_thd=None,
        short_image_path=True
):
    """Load the YOLO detections that are saved in json format to object_detection dict.
    The result can be directly used for object tracking, we do not remove empty detections.
    :param json_detection_file: one file include the detections of one video
    :param detection_dict: if not none, continue to add the object_detection to the dict, this
        is useful for ensemble of multiple detections
    :param short_image_path: if true, only save the last two levels of the image path,
        i.e., 3440/0000.jpg
    :param score_thd: remove the low score object_detection, score < score_thd
    :param nms_thd: non maximum suppression threshold
    :return:
        object_detection dict with the format of
            [class_idx, xmin, ymin, xmax, ymax, score]
        A lot of public codes use the format of [class_idx, score, xmin, ymin, xmax, ymax],
        If we do need the public format, just conduct transformation.
    """

    with open(json_detection_file) as json_file:
        json_detection_data = json.load(json_file)

    if detection_dict is None:
        detection_dict = {}

    for frame_det in json_detection_data:  # detection_idx,
        frame_det_dict = parse_yolov4_frame_det(frame_det)
        image_path = frame_det_dict["image_path"]
        if short_image_path:
            image_path = image_path_last_two_level(image_path)

        detection_dict[image_path] = frame_det_dict["det_bbox_list"]

    if score_thd is not None:
        filter_out_low_score_detection_dict(detection_dict, score_thd)

    if nms_thd is not None:
        detection_dict = nms_detection_dict(detection_dict, nms_thd)
    return detection_dict
