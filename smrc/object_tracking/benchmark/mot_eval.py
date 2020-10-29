import sys
sys.path.append("../../")
sys.path.append("../../../")

import smrc.utils
from smrc.object_tracking.benchmark.utils import *
from smrc.object_tracking.benchmark.data_config import MOT_DataRootDir
from smrc.object_tracking.ahc_ete import AHC_ETE

inf_value = float("inf")


class MOTEval:
    # MOT detection format, tlwh : array_like
    #         Bounding box in format `(x1, y1, w, h)`.
    def __init__(self):
        self.data_root_dir = MOT_DataRootDir
        # self.name = 'MOTChallenge'

        self.mot_detection_npy_dir = {
            'MOT15/train': 'MOT15_train',
            'MOT16/train': 'MOT16_train',
            # 'MOT17/train': 'MOT17_train',
            # 'MOT20/train': 'MOT20_train',
            # 'MOT15/test': 'MOT15_test',
            # 'MOT16/test': 'MOT16_test',
            # 'MOT17/test': 'MOT17_test',
            # 'MOT20/test': 'MOT20_test'
        }

    def load_all_det_for_one_sequence_from_npy_file(self, mot_root_dir, sequence_dir):
        return self.load_det_for_one_sequence_from_npy_file(
            mot_root_dir, sequence_dir,
            min_confidence=-inf_value, nms_thd=1.0, min_detection_height=0
        )

    def load_det_for_one_sequence_from_npy_file(
            self, mot_root_dir, sequence_dir, min_confidence,
            nms_thd, min_detection_height=0):

        detection_file = os.path.join(
            self.data_root_dir,
            'resources/detections',
            self.mot_detection_npy_dir[mot_root_dir],
            sequence_dir + '.npy')
        video_detection_list, video_feature_list = load_detection_with_deep_sort_feature(
            image_dir=os.path.join(self.data_root_dir, mot_root_dir, sequence_dir),
            detection_file=detection_file,
            min_confidence=min_confidence, nms_thd=nms_thd,
            min_detection_height=min_detection_height
        )
        return video_detection_list, video_feature_list

    def evaluate_AHC_ETE(self, mot_root_dir, expert_team_config):
        # mot_root_dir = 'MOT16/train'
        mot_root_full_dir = os.path.join(self.data_root_dir, mot_root_dir)
        sequence_dir_list = smrc.utils.get_dir_list_in_directory(
            mot_root_full_dir
        )
        for sequence_dir in sequence_dir_list:
            video_detection_list, video_feature_list = self.load_all_det_for_one_sequence_from_npy_file(
                mot_root_dir=mot_root_dir, sequence_dir=sequence_dir
            )
            num_detections = [len(detection_list) for image_path, detection_list in video_detection_list]
            print(f'num_detections={sum(num_detections)} ...')

            tracker = AHC_ETE()
            tracker.offline_tracking(
                video_detection_list=video_detection_list,
                video_feature_list=video_feature_list,
                expert_team_config=expert_team_config,
                benchmark_data_root_dir=self.data_root_dir,
                visualization_sequence_dir=os.path.join(mot_root_dir, sequence_dir)
            )
            tracker.clusters = tracker.sorted_clusters_based_on_length(tracker.clusters)
            tracker.delete_cluster_with_length_thd(clusters=tracker.clusters, track_min_length_thd=3)

            mot_result_dir = os.path.dirname(mot_root_dir)
            smrc.utils.generate_dir_if_not_exist(mot_result_dir)
            tracker.save_tracking_result_mot_format(
                os.path.join(mot_result_dir, sequence_dir + '.txt')
            )


if __name__ == "__main__":
    mot = MOTEval()

    from smrc.object_tracking.benchmark.expertconfig_ahc_ete import ExpertTeam
    mot.evaluate_AHC_ETE('MOT16/train', expert_team_config=ExpertTeam)
    mot.evaluate_AHC_ETE('MOT15/train', expert_team_config=ExpertTeam)
