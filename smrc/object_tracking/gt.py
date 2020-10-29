from object_tracking.visualize import Visualization


class GroundTruth(Visualization):
    def __init__(
            self, video_detections, clusters, frame_detections,
            image_path_list
    ):
        # we do not need to initialize DataHub as all the members are reinitialized below, but it's fine to do so.
        super().__init__()

        self.video_detected_bbox_all = video_detections
        self.clusters = clusters
        self.frame_dets = frame_detections
        self.IMAGE_PATH_LIST = image_path_list
        self.LAST_IMAGE_INDEX = len(self.IMAGE_PATH_LIST) - 1

        self.bbox_cluster_IDs = {}
        self._assign_cluster_id_to_global_bbox_idx()

