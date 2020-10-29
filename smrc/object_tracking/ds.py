from tqdm import tqdm
import cv2
import numpy as np
import os

from .tracker import TrackerSMRC, DataHub
import smrc.utils

from .deep_sort import nn_matching, preprocessing
from .deep_sort.detection import Detection
from .deep_sort.tracker import Tracker
from .deep_sort import generate_detections as gdet


class DeepSort(DataHub):
    """Generating features from the detection results. The tracking is not conducted in this class.
    """
    def __init__(self):
        super().__init__()
        model_root_dir = smrc.utils.dir_name_up_n_levels(
            file_abspath=os.path.abspath(__file__),
            n=2
        )
        self.model_filename = os.path.join(model_root_dir, 'model_data/mars-small128.pb')
        # model_filename = 'smrc/object_tracking/deep_sort/model_data/mars-small128.pb'
        assert os.path.isfile(self.model_filename)
        self.encoder = gdet.create_box_encoder(self.model_filename, batch_size=1)
        self.deep_sort_feature_available = False

    def encode_features_for_frame_det(self, color_frame, bbox_list):
        box_rects = smrc.utils.transfer_bbox_list_x1y1wh_format(
            bbox_list, with_class_index=False
        )
        # print("box_num",len(boxs))
        features = self.encoder(color_frame, box_rects)

        # # score to 1.0 here).
        # detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(box_rects, features)]
        return features

    def encode_features_for_video_det(self):
        pbar = tqdm(enumerate(self.IMAGE_PATH_LIST))
        for image_id, img_path in pbar:
            points_current_image = self.frame_dets[img_path]
            pbar.set_description(
                f"Generating feature vectors for "
                f"image_id = {image_id}/{len(self.IMAGE_PATH_LIST)}, {len(points_current_image)} boxes ...")

            if len(points_current_image) == 0:
                continue

            # the order of global_bbox_ids and bbox_list are identical
            global_bbox_ids = [x for x in points_current_image]
            bbox_list = [self.get_single_bbox(x) for x in points_current_image]
            tmp_img = cv2.imread(img_path)
            features = self.encode_features_for_frame_det(tmp_img, bbox_list)

            # score to 1.0 here).
            for feature, global_bbox_id in zip(features, global_bbox_ids):
                self.video_detected_bbox_all[global_bbox_id]['feature'] = feature  # array

        self.deep_sort_feature_available = True

    def get_feature_dict_ready(self):
        if not self.deep_sort_feature_available:
            self.encode_features_for_video_det()


class DeepSortOnline(TrackerSMRC, DeepSort):
    """Conducting online deep sort based tracking.

    """
    def __init__(self):
        super(TrackerSMRC, self).__init__()
        super(DeepSort, self).__init__()

        # model_filename=self.model_filename
        self.Tracker_Params = dict(
            max_cosine_distance=0.3,
            nn_budget=None,
            nms_max_overlap=1.0,
            resulting_tracks_image_dir=None,
            resulting_tracks_txt_file=None
        )

    def online_tracking(self, video_detection_list, **kwargs):
        self.init_tracking_tool(video_detection_list, **kwargs)
        if len(self.video_detected_bbox_all) == 0:
            print('Tracking impossible, no detection loaded, return ...')
            return

        results = self.tracking_main()

        if self.Tracker_Params['resulting_tracks_image_dir'] is not None:
            resulting_image_dir = self.Tracker_Params['resulting_tracks_image_dir']
            self.online_tracking_results_to_video(results, resulting_image_dir)

        if self.Tracker_Params['resulting_tracks_txt_file'] is not None:
            result_file_name = self.Tracker_Params['resulting_tracks_txt_file']
            with open(result_file_name, 'w') as new_file:
                # assert isinstance(results, list)
                for row in results:
                    # image_id, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]]
                    row = self.to_int_list(row)
                    txt_line = ', '.join(map(str, row))
                    new_file.write(txt_line + '\n')

    @staticmethod
    def to_int_list(my_list):
        return list(map(int, my_list))

    def online_tracking_results_to_video(
            self, tracked_bbox_list, resulting_dir,
            blank_bg=False,
            fps=30
        ):
        """Generating videos for deep sort online tracking results
        :param tracked_bbox_list: [
            [image_id, track_id, bbox[0], bbox[1], bbox[2], bbox[3]],
            ...
        ]
        :param resulting_dir:
        :param blank_bg:
        :param fps:
        :return:
        """
        tracker_name = self.__class__.__name__

        print(f'Saving object_tracking results to {os.path.abspath(resulting_dir)} ...')
        smrc.utils.generate_dir_if_not_exist(resulting_dir)

        assert len(tracked_bbox_list) > 0, 'There should be more than one cluster ...'
        bbox_to_plot = [[] for image_id, _ in enumerate(self.IMAGE_PATH_LIST)]

        # image_id, track_id, bbox[0], bbox[1], bbox[2], bbox[3]]
        max_track_id = 0
        for image_id, track_id, x1, y1, x2, y2 in tracked_bbox_list:
            bbox_to_plot[image_id].append(self.to_int_list([track_id, x1, y1, x2, y2]))
            max_track_id = max(max_track_id, track_id)

        object_colors = smrc.utils.color.unique_colors(max_track_id + 1)
        height, width = smrc.utils.get_image_size(
            self.IMAGE_PATH_LIST[0]
        )

        if width > 800:
            line_thickness = 2
            font_scale = 0.8
        else:
            line_thickness = 1
            font_scale = 0.6

        for image_id, image_path in tqdm(enumerate(self.IMAGE_PATH_LIST)):
            if blank_bg:
                tmp_img = smrc.utils.generate_blank_image(height, width)
            else:
                tmp_img = cv2.imread(image_path)
            # display the tracker name on the image
            smrc.utils.draw.put_text_on_image(
                tmp_img, text_content=tracker_name,
                thickness=4, font_scale=2
            )
            # plot tracking result if any
            if len(bbox_to_plot[image_id]) > 0:
                for object_id, xmin, ymin, xmax, ymax in bbox_to_plot[image_id]:
                    object_color = object_colors[object_id]
                    text = 'obj ' + str(object_id)
                    smrc.utils.draw.draw_bbox_legend(
                        tmp_img=tmp_img, text_content=text,
                        location_to_draw=(xmin, ymin),
                        text_shadow_color=object_color,
                        text_color=(0, 0, 0),
                        font_scale=font_scale,
                        line_thickness=line_thickness
                    )
                    cv2.rectangle(
                        tmp_img, (xmin, ymin), (xmax, ymax),
                        object_color, line_thickness
                    )

            image_path_new = os.path.join(
                    resulting_dir, os.path.basename(image_path)
            )
            # print(f'Saving object_tracking results to {image_path_new}')
            cv2.imwrite(image_path_new, tmp_img)

        pathIn, pathOut = resulting_dir, resulting_dir + '.avi'
        print(f'Generating object_tracking results to {os.path.abspath(pathOut)} ...')
        smrc.utils.convert_frames_to_video(pathIn, pathOut, fps)

    def tracking_main(self):
        """
        :rtype: clusters
        """
        # deep_sort, what model ?
        model_filename = self.model_filename
        encoder = gdet.create_box_encoder(model_filename, batch_size=1)

        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine",
            self.Tracker_Params['max_cosine_distance'],
            self.Tracker_Params['nn_budget']
        )
        tracker = Tracker(metric)
        # Store results.
        tracking_results = []  # the format to be directly written in txt files.
        h, w = None, None
        for image_id, image_path in tqdm(enumerate(self.IMAGE_PATH_LIST)):
            if len(self.frame_dets[image_path]) == 0:
                detections = []
            else:
                global_bbox_id_list = self.frame_dets[image_path]

                bbox_list = self.get_bbox_list_for_cluster(global_bbox_id_list)
                scores = self.get_score_list_for_cluster(global_bbox_id_list)
                boxes = smrc.utils.transfer_bbox_list_x1y1wh_format(
                    bbox_list, with_class_index=False
                )  # boxes_transferred
                # need to check if cv2.imread is same with video.capture.read()
                frame = cv2.imread(image_path)
                features = encoder(frame, boxes)
                h, w = frame.shape[:2]
                # score to 1.0 here).
                detections = [
                    Detection(bbox, score, feature)
                    for bbox, feature, score in
                    zip(boxes, features, scores)
                ]

                # Run non-maxima suppression.
                boxes = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                indices = preprocessing.non_max_suppression(
                    boxes, self.Tracker_Params['nms_max_overlap'], scores)
                detections = [detections[i] for i in indices]

            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:  # original
                    continue
                #############################################
                # to smrc format
                #############################################
                bbox = track.to_tlbr()
                # track_id of deep sort is 1-index, we need to transfer it to 0-index
                # Thus we have track.track_id-1
                tracking_results.append([
                    image_id, track.track_id-1,
                    max(bbox[0], 0), max(bbox[1], 0),  # make sure x1, y1 >= 0
                    min(w, bbox[2]), min(h, bbox[3])]  # make sure x2 < w, y2 < h
                )

        return tracking_results


