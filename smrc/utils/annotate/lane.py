import cv2
import os
import sys
import numpy as np


class LaneObj:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)

        # the following code is wrong, we should not save the p1, p2 this way
        # # all the coordinates are automatically maintained
        # self.x1 = int(min(x1, x2))
        # self.y1 = int(min(y1, y2))
        # self.x2 = int(max(x1, x2))
        # self.y2 = int(max(y1, y2))

    def length(self):
        # np.linalg.norm(np.asarray([3,4]))= 5.0
        return np.linalg.norm(
            np.asarray([self.x1, self.y1]) - np.asarray([self.x2, self.y2])
        )
    
    def to_list(self):
        return [self.x1, self.y1, self.x2, self.y2]

    def to_int_list(self):
        return list(map(round, self.to_list()))

    def to_txt_line(self):
        return ' '.join(map(str, self.to_int_list()))

    @classmethod
    def from_p1_p2(cls, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return cls(x1, y1, x2, y2)


def load_lane_from_txt_file(ann_path):
    # If there is no annotation, there may be no corresponding txt files.
    # So we do not need to require that the annotation file exists.

    lane_list = []
    if os.path.isfile(ann_path):
        with open(ann_path, 'r') as old_file:
            lines = old_file.readlines()
        old_file.close()

        # print('lines = ',lines)
        for line in lines:
            result = line.split(' ')

            # # the data format in line (or txt file) should be int type, 0-index.
            # # we transfer them to int again even they are already in int format (just in case they are not)
            # lane = [int(result[0]), int(result[1]), int(result[2]), int(result[3])]
            lane = LaneObj(x1=int(result[0]), y1=int(result[1]), x2=int(result[2]), y2=int(result[3]))
            lane_list.append(lane)
    return lane_list


def save_lane_to_txt_file(ann_path, lane_list):
    with open(ann_path, 'w') as new_file:
        for lane in lane_list:
            txt_line = lane.to_txt_line()
            # we need to add '\n'(newline), otherwise, all the bboxes will be in one line and not able to be recognized.
            new_file.write(txt_line + '\n')
    new_file.close()


def remove_lane_from_txt_file(ann_path, lane):
    lane_list_old = load_lane_from_txt_file(ann_path=ann_path)

    assert lane in lane_list_old
    # if lane in lane_list_old:
    lane_list_old.remove(lane)
    save_lane_to_txt_file(ann_path=ann_path, lane_list=lane_list_old)


def add_lane_list_to_txt_file_incrementally(ann_path, lane_list):
    lane_list_old = load_lane_from_txt_file(ann_path=ann_path)

    with open(ann_path, 'a') as new_file:
        for lane in lane_list:
            # if this bbox is already in the file, then not add it.
            if lane in lane_list_old:
                continue
            else:
                # add the new lane
                txt_line = lane.to_txt_line()
                # we need to add '\n'(newline), otherwise, all the lanes will be in one line and not able to be recognized.
                new_file.write(txt_line + '\n')

                # update the lane_list_old so that even there is no redundant lane
                lane_list_old.append(lane)
    new_file.close()


def add_singe_lane_to_txt_file_incrementally(ann_path, lane):
    add_lane_list_to_txt_file_incrementally(ann_path=ann_path, lane_list=[lane])


def edit_lane_in_txt_file(ann_path, lane_old, lane_new):
    lane_list = load_lane_from_txt_file(ann_path=ann_path)

    assert lane_old in lane_list
    # if lane in lane_list_old:
    lane_list.remove(lane_old)
    lane_list.append(lane_new)
    save_lane_to_txt_file(ann_path=ann_path, lane_list=lane_list)
