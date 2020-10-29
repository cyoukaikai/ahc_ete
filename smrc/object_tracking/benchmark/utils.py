from tqdm import tqdm
import numpy as np
import os
import cv2

from object_tracking.deep_sort.detection import Detection
from object_tracking.deep_sort import preprocessing


# from https://github.com/nwojke/deep_sort/blob/master/deep_sort_app.py
def gather_sequence_info(sequence_dir, detection_file):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    detections = None
    if detection_file is not None:
        detections = np.load(detection_file)
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,  # dict
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info


# from https://github.com/nwojke/deep_sort/blob/master/deep_sort_app.py
def create_detections(detection_mat, frame_idx, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        # 3,-1,  378.26,395.41,84.742,256.23,  3.2163,  -1,-1,-1
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list


def load_detection_with_deep_sort_feature(
        image_dir, detection_file, min_confidence, nms_thd, min_detection_height=0):
    video_detections, features = load_mot_detections_with_features(
        sequence_dir=image_dir, detection_file=detection_file,
        min_confidence=min_confidence, nms_max_overlap=nms_thd,
        min_detection_height=min_detection_height
    )
    return video_detections, features


def load_mot_detections_with_features(
        sequence_dir, detection_file, min_confidence,
        nms_max_overlap, min_detection_height
        ):
    """Load detection and features from .npy files.
    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    """
    seq_info = gather_sequence_info(sequence_dir, detection_file)
    video_detections = []
    video_features = []

    # the following sentence can get all detections and their features loaded, but the orders of the sequence is lost
    # as the seq_info["image_filenames"] is a dict, it's saved based on the image path sequence (temporal order)
    # for frame_idx, image_path in seq_info["image_filenames"].items():
    pbar = tqdm(range(seq_info["min_frame_idx"], seq_info["max_frame_idx"] + 1))
    for frame_idx in pbar:
        if frame_idx not in seq_info["image_filenames"]:
            continue
        pbar.set_description("Processing frame %05d/%05d ..." % (frame_idx, seq_info["max_frame_idx"]))
        # print("Processing frame %05d" % frame_idx)
        # image_path = seq_info["image_filenames"][frame_idx]

        # Load image and generate detections.
        detections = create_detections(
            seq_info["detections"], frame_idx, min_detection_height)
        detections = [d for d in detections if d.confidence >= min_confidence]

        # Run non-maximum suppression for the original data format of deep sort
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        feature = np.array([d.feature for d in detections])

        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)

        # bbox[:, 2:4] += bbox[:, 0:2]  # x1, y1, w, h -> x1, y1, x2, y2

        # transfer the bbox format, #bbox, confidence, feature
        boxes = np.array([d.to_tlbr() for d in detections])  # np.array([d.tlwh for d in detections])
        detections = [[0] + list(boxes[i]) + [scores[i]] for i in indices]  # fake class id = 0
        # dets.append([class_idx, bb[0], bb[1], bb[2], bb[3], s])
        video_detections.append([
            seq_info["image_filenames"][frame_idx],
            detections
        ])
        video_features.append(feature)

    return video_detections, video_features


