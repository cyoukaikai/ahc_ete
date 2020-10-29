import os
# import sys
# import json
# import numpy as np
# from tqdm import tqdm

import smrc.utils

######################################################################
# import object_detection results to the annotation tool
# the code need to be improved according to the code above in order to
# take the advantage of the above code.
######################################################################


class ImportDetection:
    # Transfer YOLO smrc json format to annotation format [class_label, x1, y1, x2, y2]
    # We conduct non_max_suppression and remove low score object_detection.

    # To use this tool, make sure the detections are put in json_file_dir (default dir,'detection_json_file')
    # The imported results will be put in label_dir (default dir, 'labels')
    def __init__(self,
                 json_file_dir='detection_json_file',  # the dir where the json files are
                 label_dir=None,  # the dir where the output labels are saved
                 score_thd=0.25,  # threshold value for confidence level
                 nms_thd=0.5,  # IoU threshold for non_max_suppression
                 source_class_list_file=None,
                 target_class_list_file=None
                 ):
        self.json_file_dir = json_file_dir
        if label_dir is None:
            self.label_dir = json_file_dir + '_labels' + str(score_thd) + '_nms' + str(nms_thd)
            # + '_labels-score-thd' + str(score_thd) + '_nms-thd' + str(nms_thd)
        else:
            self.label_dir = label_dir

        smrc.utils.generate_dir_if_not_exist(self.label_dir)

        self.score_thd = score_thd  # threshold for the posterior probablity of the detections
        self.nms_thd = nms_thd  # threshold for non_max_suppression
        self.source_class_list_file = source_class_list_file
        self.target_class_list_file = target_class_list_file

        print("====================================================================")
        print(f'self.json_file_dir = {os.path.abspath(self.json_file_dir)} ...')
        print(f'self.label_dir = {os.path.abspath(self.label_dir)} ...')
        print(f'score_thd = {self.score_thd}, non_max_suppression_thd = {self.nms_thd}')
        print("====================================================================")
        # import the object_detection
        self.import_detection_to_annotation_tool()

    @staticmethod
    def image_path_to_ann_path(image_path, dir_name, label_dir):
        str_fo_find = os.path.sep + dir_name + os.path.sep
        image_root_dir = image_path[0:image_path.rfind(str_fo_find)]
        img_file_name = image_path.replace(image_root_dir, label_dir, 1)
        pre_path, img_ext = os.path.splitext(img_file_name)
        ann_path = img_file_name.replace(img_ext, '.txt', 1)
        return ann_path

    @staticmethod
    def det_list_to_bbox_list(image_det):
        return [det[:5] for det in image_det]

    @staticmethod
    def filter_det_dict_by_class_label(detection_dict, label_ref):
        """
        The classes that are out of interest are marked as None.
        :param detection_dict: the detections we want to transfer
        :param label_ref: {
            0: 0,
            1: None,
            ...,
            80: 3
        }
        :return:
        """
        for key, dets in detection_dict.items():
            detction_list_new = []
            for det in dets:
                class_id = det[0]
                if class_id not in label_ref:
                    print(f'Note class_id {class_id} not in label_ref, this may be due to the detector ... ')
                else:
                    if label_ref[class_id] is not None:
                        det[0] = label_ref[class_id]  # mapping the class id to new id
                        detction_list_new.append(det)
            detection_dict[key] = detction_list_new
        return detection_dict

    @staticmethod
    def build_label_refers(source_class_list_file, target_class_list_file):
        # load the target class list
        target_class_list = smrc.utils.load_class_list_from_file(target_class_list_file)
        print(f'Loaded target_class_list = {target_class_list} ...')

        source_class_list = smrc.utils.load_class_list_from_file(source_class_list_file)
        print(f'Loaded {source_class_list} class list = {source_class_list} ...')

        label_ref = {}
        # if coco_class_idx < len(self.COCO_class_list):
        for class_idx, class_name in enumerate(source_class_list):
            if class_name in target_class_list:
                target_class_idx = target_class_list.index(class_name)
                label_ref[class_idx] = target_class_idx

                # display the built label_ref
                print('{:20s}: {:4d} ---> {:20s}: {:4d}'.format(
                    class_name, class_idx, target_class_list[target_class_idx], target_class_idx)
                )
            else:
                label_ref[class_idx] = None
                print('{:20s}: {:3d} ---> {:20s}'.format(
                    class_name, class_idx, 'None')
                )

        return label_ref

    def import_detection_to_annotation_tool(self):
        # load only the directory name not the full path (True)
        # json_file_list = smrc.line.get_smrc_json_file_list(
        #     self.json_file_dir, only_local_name=True
        # )
        json_file_list = smrc.utils.get_json_file_list(
            self.json_file_dir, only_local_name=True
        )

        label_ref = None
        if self.target_class_list_file is not None and \
                self.source_class_list_file is not None and \
                self.source_class_list_file != self.target_class_list_file:
            label_ref = self.build_label_refers(
                source_class_list_file=self.source_class_list_file,
                target_class_list_file=self.target_class_list_file
            )

        # dir_list = smrc.line.get_json_dir_list(self.json_file_dir)
        for dir_idx, json_file_name in enumerate(json_file_list):
            dir_name = smrc.utils.extract_smrc_json_dir(json_file_name)
            dir_path = os.path.join(self.label_dir, dir_name)

            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            detection_file = os.path.join(self.json_file_dir, json_file_name)
            print(f'{dir_idx + 1}/{len(json_file_list)}, Handling {detection_file} ...')

            # key is the image path, value is the bbox_list with scores
            detection_dict = smrc.utils.load_json_detection_to_dict(
                json_detection_file=detection_file,
                score_thd=self.score_thd, nms_thd=self.nms_thd,
                short_image_path=False
            )

            if label_ref is not None:
                detection_dict = self.filter_det_dict_by_class_label(detection_dict, label_ref)

            for image_path in detection_dict:
                ann_path = self.image_path_to_ann_path(
                    image_path, dir_name, self.label_dir
                )
                bbox_list = self.det_list_to_bbox_list(detection_dict[image_path])
                smrc.utils.save_bbox_to_file(ann_path, bbox_list)
