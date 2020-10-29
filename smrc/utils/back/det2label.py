import os
import sys
import json
import numpy as np

import smrc.utils
# import smrc.line.detection_process

######################################################################
# import object_detection results to the annotation tool
# the code need to be improved according to the code above in order to
# take the advantage of the above code.
######################################################################


class ImportDetection:
    # input, object_detection dir
    # output, annotation in smrc format (class_label, x1, y1, x2, y2)
    # we will conduct non_max_suppression inside the tool

    # to use this tool, make sure the detections are put in json_file_dir (default dir,'detection_json_file')
    # the imported results will be put in label_dir (default dir, 'labels')
    def __init__(self,
                 json_file_dir='detection_json_file',  # the dir where the json files are
                 label_dir=None,  # the dir where the output labels are saved
                 score_thd=0.25,  # threshold value for confidence level
                 detection_format='smrc',
                 class_list_file='smrc.names',
                 non_max_suppression_thd=0.65  # IoU threshold for non_max_suppression
                 ):
        self.json_file_dir = json_file_dir
        if label_dir is None:
            self.label_dir = os.path.join(json_file_dir, 'labels')  # output
        else:
            self.label_dir = label_dir

        smrc.utils.generate_dir_if_not_exist(self.label_dir)

        self.score_thd = score_thd  # threshold for the posterior probablity of the detections
        self.nms_thd = non_max_suppression_thd  # threshold for non_max_suppression
        print(f'score_thd = {self.score_thd}, non_max_suppression_thd = {self.nms_thd}')

        ## initialize class list
        self.class_list_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), class_list_file
        )
        self.class_list = None
        self.COCO_class_list = None
        self.COCO_class_name_file = 'coco.names'

        # tt = os.path.curdir()
        with open(self.class_list_file) as f:
            CLASS_LIST = list(smrc.utils.non_blank_lines(f))
        f.close()  # close the file
        # print(self.CLASS_LIST)
        self.class_list = CLASS_LIST
        # self.class_list = CLASS_LIST[ 0:len(self.class_index_list)]
        print(self.class_list)

        # import the object_detection
        self.import_detection_to_annotation_tool(detection_format)

    def import_detection_to_annotation_tool(self, detection_format='smrc'):
        # print('self.json_file_dir', self.json_file_dir)

        # load only the directory name not the full path (True)
        json_file_list = smrc.utils.get_smrc_json_file_list(self.json_file_dir, True)

        # dir_list_to_process = dir_list_to_process[0:2]
        print('To load {} json files with the format smrc_dirname.json.'.format(len(json_file_list)))
        # print(json_file_list)

        # load the object_detection results in self.json_file_dir with threshold self.score_thd
        # to self.label_dir, conduct non_max_suppression with threshold self.nms_thd
        # dir_list_to_process controls which directory to import

        self.load_detection_from_json_file(json_file_list,
                                           self.score_thd,
                                           self.json_file_dir,
                                           self.label_dir,
                                           self.nms_thd,
                                           detection_format
                                           )

    # load the detections as the starting point for the annotation work
    # format of detections:  {'image_path': '/home/smrc/Data/1000videos/3440/0000.jpg',
    # 'category_id': 2, 'bbox': [275, 187, 78, 53], 'score': 0.984684}
    def load_detection_from_json_file(self, json_file_list, score_thd, json_file_dir,
                                      resulting_dir, overlap_thd, detection_format='smrc'):
        for dir_idx, json_file_name in enumerate(json_file_list):
            dir_name = json_file_name[
                       json_file_name.find('smrc_') + len('smrc_'): json_file_name.find('.json')
                       ]

            dir_path = os.path.join(resulting_dir, dir_name)
            print('Importing {} to {}, {}/{}'.format(
                os.path.join(self.json_file_dir, dir_name),
                dir_path, dir_idx + 1, len(json_file_list)
            )
            )

            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            # load the object_detection file
            detection_file = os.path.join(json_file_dir, json_file_name)
            with open(detection_file) as json_file:
                detection_data = json.load(json_file)

            if detection_format == 'smrc':
                detection_dict = self.process_json_detection_to_bbox_list(
                    detection_data, dir_name,
                    resulting_dir,
                    score_thd
                )
            elif detection_format == 'coco':
                detection_dict = self.process_coco_json_detection_to_bbox_list(
                    detection_data, dir_name,
                    resulting_dir,
                    score_thd
                )
            else:
                print(f'Unkown detection format: {detection_format}')
                sys.exit(0)

            self.save_json_detection(detection_dict, overlap_thd)

    @staticmethod
    def process_json_detection_to_bbox_list(
            loaded_json_detection_data,
            dir_name, resulting_dir, score_thd
    ):
        # {'image_path': '/home/smrc/Data/1000videos/3440/0000.jpg', 'category_id': 2,
        # 'bbox': [275, 187, 78, 53], 'score': 0.984684}

        # all the detections regarding this directory are saved here
        detection_dict = {}
        for detection in loaded_json_detection_data:  # detection_idx,
            # load the image name
            image_path = detection['image_path']
            class_idx = detection['category_id']
            xmin, ymin, xmax, ymax = detection['bbox']
            score = detection['score']

            dir_old = image_path[0:image_path.find(dir_name) - 1]
            img_file_name = image_path.replace(dir_old, resulting_dir, 1)
            pre_path, img_ext = os.path.splitext(img_file_name)
            ann_path = img_file_name.replace(img_ext, '.txt', 1)

            # print('ann_path = %s, class_idx = %d, xmin = %d, ymin =%d, xmax = %d, ymax =%d, score = %4.1f' % (
            #     (ann_path, class_idx, xmin, ymin, xmax, ymax, score))
            # )
            if score >= score_thd:
                if ann_path in detection_dict:
                    detection_dict[ann_path].append([class_idx, xmin, ymin, xmax, ymax, score])
                else:
                    detection_dict[ann_path] = [
                        [class_idx, xmin, ymin, xmax, ymax, score]
                    ]

        return detection_dict

    @staticmethod
    def save_json_detection(detection_dict, overlap_thd):
        # input (dict, format: dict{ann_path}= raw_bbox )
        # conduct non maximum suppression
        # and save the results to annotation file
        for ann_path, raw_bbox in detection_dict.items():

            raw_bbox_array = np.array(raw_bbox)
            boxes, scores = raw_bbox_array[:, 1:5], raw_bbox_array[:, 5:].flatten()

            if len(raw_bbox) > 1:
                selected = smrc.utils.non_max_suppression(boxes, scores, overlap_thd)
                bbox_processed = raw_bbox_array[selected, 0:5].astype("int").tolist()
            else:
                bbox_processed = raw_bbox_array[:, 0:5].astype("int").tolist()

            smrc.utils.save_bbox_to_file(ann_path, bbox_processed)

    def process_coco_json_detection_to_bbox_list(self,
                                                 loaded_json_detection_data,
                                                 dir_name, resulting_dir, score_thd
                                                 ):
        # {'image_path': '/home/smrc/Data/1000videos/3440/0000.jpg', 'category_id': 2,
        # 'bbox': [275, 187, 78, 53], 'score': 0.984684}

        # all the detections regarding this directory are saved here
        with open(self.COCO_class_name_file) as f:
            self.COCO_class_list = list(smrc.utils.non_blank_lines(f))
        f.close()  # close the file

        detection_dict = {}
        for detection in loaded_json_detection_data:  # detection_idx,
            # load the image name
            image_path = detection['image_path']
            coco_class_idx = detection['category_id']
            xmin, ymin, xmax, ymax = detection['bbox']
            score = detection['score']

            dir_old = image_path[0:image_path.find(dir_name) - 1]
            img_file_name = image_path.replace(dir_old, resulting_dir, 1)
            pre_path, img_ext = os.path.splitext(img_file_name)
            ann_path = img_file_name.replace(img_ext, '.txt', 1)

            # print('ann_path = %s, class_idx = %d, xmin = %d, ymin =%d, xmax = %d, ymax =%d, score = %4.1f' % (
            #     (ann_path, class_idx, xmin, ymin, xmax, ymax, score))
            # )
            # {"image_path":"/home/smrc/Data/ML_Samples_20181109b/105189/0139.jpg", "category_id":80, "bbox":[86, 348, 425, 397], "score":0.007549},
            # do not know why
            if coco_class_idx < len(self.COCO_class_list):
                coco_class_name = self.COCO_class_list[coco_class_idx]
                if score >= score_thd and coco_class_name in self.class_list:
                    class_idx = self.class_list.index(coco_class_name)
                    if ann_path in detection_dict:
                        detection_dict[ann_path].append([class_idx, xmin, ymin, xmax, ymax, score])
                    else:
                        detection_dict[ann_path] = [
                            [class_idx, xmin, ymin, xmax, ymax, score]
                        ]
        return detection_dict
