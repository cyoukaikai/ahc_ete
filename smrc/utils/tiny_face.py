
# class TinyFaceDetectionProcess:
#
#     def __init__(self, score_position):
#         # object_detection format of the detected bbox
#         self.ConfidenceScorePosition = {
#             'second': 1,  # [class_idx, x1, y1, x2, y2, score]
#             'last': -1  # [class_idx, score, x1, y1, x2, y2]
#         }
#
#         assert score_position in self.ConfidenceScorePosition
#         self.score_position = score_position
#         # self.det_idx = self.ConfidenceScorePosition[score_position]
#


# def test_face_extraction():
#     image_dir = 'Truck-sampleData114videos'  # 'image-inside-car'
#     label_dir = 'resutls'  # resutls
#     score_thd = 0.5
#     extract_raw_detection_to_bbox_list(image_dir, label_dir, score_thd, score_position='second')
